"""Flax NNX ports of DART's MLD denoisers."""

from __future__ import annotations

import argparse
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .torch_compatible import TorchLayerNorm, TorchLinear, TorchMultiheadAttention


def _gelu_exact(x):
    return 0.5 * x * (1.0 + jax.lax.erf(x / jnp.sqrt(2.0)))


def _activation(name: str):
    if name == "tanh":
        return jnp.tanh
    if name == "relu":
        return jax.nn.relu
    if name == "sigmoid":
        return jax.nn.sigmoid
    if name == "gelu":
        return _gelu_exact
    if name == "lrelu":
        return lambda x: jax.nn.leaky_relu(x, negative_slope=0.01)
    raise ValueError(f"Unsupported activation {name}.")


class PositionalEncoding(nnx.Module):
    """Torch ``PositionalEncoding`` buffer layout, ``[max_len, 1, d_model]``."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        position = jnp.arange(max_len, dtype=jnp.float32)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, d_model, 2, dtype=jnp.float32)
            * (-jnp.log(10000.0) / d_model)
        )
        pe = jnp.zeros((max_len, d_model), dtype=jnp.float32)
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term[: pe[:, 1::2].shape[-1]]))
        self.pe = nnx.Variable(pe[:, None, :])
        self.dropout = dropout

    def __call__(self, x):
        return x + self.pe[: x.shape[0], :, :]


class TimestepEmbedder(nnx.Module):
    def __init__(
        self,
        h_dim: int,
        sequence_pos_encoder: PositionalEncoding,
        *,
        rngs: nnx.Rngs,
    ):
        self.h_dim = h_dim
        self.sequence_pos_encoder = sequence_pos_encoder
        self.time_embed = nnx.List(
            [
                TorchLinear(h_dim, h_dim, rngs=rngs),
                TorchLinear(h_dim, h_dim, rngs=rngs),
            ]
        )

    def __call__(self, timesteps):
        x = self.sequence_pos_encoder.pe[timesteps]
        x = self.time_embed[0](x)
        x = jax.nn.silu(x)
        x = self.time_embed[1](x)
        return jnp.swapaxes(x, 0, 1)


class MLP(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        h_dims: Tuple[int, ...] = (128, 128),
        activation: str = "tanh",
        *,
        rngs: nnx.Rngs,
    ):
        self.activation_name = activation
        self.activation = _activation(activation)
        self.out_dim = h_dims[-1]
        layers = []
        in_dim_ = in_dim
        for h_dim in h_dims:
            layers.append(TorchLinear(in_dim_, h_dim, rngs=rngs))
            in_dim_ = h_dim
        self.layers = nnx.List(layers)

    def __call__(self, x):
        for fc in self.layers:
            x = self.activation(fc(x))
        return x


class MLPBlock(nnx.Module):
    def __init__(
        self,
        h_dim: int,
        out_dim: int,
        n_blocks: int,
        actfun: str = "relu",
        residual: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        self.residual = residual
        self.layers = nnx.List(
            [
                MLP(h_dim, h_dims=(h_dim, h_dim), activation=actfun, rngs=rngs)
                for _ in range(n_blocks)
            ]
        )
        self.out_fc = TorchLinear(h_dim, out_dim, rngs=rngs)

    def __call__(self, x):
        h = x
        for layer in self.layers:
            r = h if self.residual else 0
            h = layer(h) + r
        return self.out_fc(h)


class TransformerEncoderLayer(nnx.Module):
    """PyTorch ``nn.TransformerEncoderLayer`` parity subset.

    DART uses the default PyTorch settings: sequence-first tensors,
    ``norm_first=False``, no masks, and dropout disabled during parity eval.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        *,
        rngs: nnx.Rngs,
    ):
        self.self_attn = TorchMultiheadAttention(d_model, nhead, rngs=rngs)
        self.linear1 = TorchLinear(d_model, dim_feedforward, rngs=rngs)
        self.linear2 = TorchLinear(dim_feedforward, d_model, rngs=rngs)
        self.norm1 = TorchLayerNorm(d_model, rngs=rngs)
        self.norm2 = TorchLayerNorm(d_model, rngs=rngs)
        self.dropout = dropout
        self.activation = _activation(activation)

    def __call__(self, src):
        src_batch = jnp.swapaxes(src, 0, 1)
        src2 = self.self_attn(src_batch, src_batch, src_batch)
        src2 = jnp.swapaxes(src2, 0, 1)
        src = self.norm1(src + src2)
        src2 = self.linear2(self.activation(self.linear1(src)))
        return self.norm2(src + src2)


class TransformerEncoder(nnx.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        num_layers: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.layers = nnx.List(
            [
                TransformerEncoderLayer(
                    d_model,
                    nhead,
                    dim_feedforward,
                    dropout,
                    activation,
                    rngs=rngs,
                )
                for _ in range(num_layers)
            ]
        )

    def __call__(self, src):
        output = src
        for layer in self.layers:
            output = layer(output)
        return output


def _require_integer_timesteps(timesteps, module_name: str):
    timesteps = jnp.asarray(timesteps)
    if not jnp.issubdtype(timesteps.dtype, jnp.integer):
        raise TypeError(f"{module_name} timesteps must be integer indices.")
    return timesteps.astype(jnp.int32)


class _DenoiserConditioningMixin:
    def mask_cond(
        self,
        cond,
        *,
        force_mask: bool = False,
        training: bool = False,
        rng: Optional[jax.Array] = None,
    ):
        if force_mask:
            return jnp.zeros_like(cond)
        if training and self.cond_mask_prob > 0.0:
            if rng is None:
                raise ValueError("mask_cond requires rng when training with cond_mask_prob.")
            mask = jax.random.bernoulli(
                rng,
                p=self.cond_mask_prob,
                shape=(cond.shape[0], 1),
            ).astype(cond.dtype)
            return cond * (1.0 - mask)
        return cond


class DenoiserMLP(_DenoiserConditioningMixin, nnx.Module):
    def __init__(
        self,
        h_dim: int = 512,
        n_blocks: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
        clip_dim: int = 512,
        history_shape: Tuple[int, int] = (2, 276),
        noise_shape: Tuple[int, int] = (1, 128),
        cond_mask_prob: float = 0.0,
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        self.h_dim = h_dim
        self.dropout = dropout
        self.n_blocks = n_blocks
        self.activation = activation
        self.history_shape = tuple(int(x) for x in history_shape)
        self.noise_shape = tuple(int(x) for x in noise_shape)
        self.clip_dim = clip_dim
        self.cond_mask_prob = float(cond_mask_prob)

        self.sequence_pos_encoder = PositionalEncoding(self.h_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(
            self.h_dim,
            self.sequence_pos_encoder,
            rngs=rngs,
        )
        self.history_dim = int(np.prod(self.history_shape))
        self.noise_dim = int(np.prod(self.noise_shape))
        input_dim = self.h_dim + self.clip_dim + self.history_dim + self.noise_dim
        self.input_project = TorchLinear(input_dim, self.h_dim, rngs=rngs)
        self.mlp = MLPBlock(
            h_dim=self.h_dim,
            out_dim=self.noise_dim,
            n_blocks=self.n_blocks,
            actfun=self.activation,
            rngs=rngs,
        )

    def __call__(
        self,
        x_t,
        timesteps,
        y,
        *,
        training: bool = False,
        rng: Optional[jax.Array] = None,
    ):
        batch_size = x_t.shape[0]
        timesteps = _require_integer_timesteps(timesteps, "DenoiserMLP")
        emb_time = self.embed_timestep(timesteps)[0]
        emb_history = jnp.reshape(
            y["history_motion_normalized"],
            (batch_size, self.history_dim),
        )
        emb_text = self.mask_cond(
            y["text_embedding"],
            force_mask=bool(y.get("uncond", False)),
            training=training,
            rng=rng,
        )
        emb_noise = jnp.reshape(x_t, (batch_size, self.noise_dim))
        input_embed = jnp.concatenate(
            (emb_time, emb_text, emb_history, emb_noise),
            axis=1,
        )
        output = self.mlp(self.input_project(input_embed))
        return jnp.reshape(output, (batch_size, *self.noise_shape))


class DenoiserTransformer(_DenoiserConditioningMixin, nnx.Module):
    def __init__(
        self,
        h_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        clip_dim: int = 512,
        history_shape: Tuple[int, int] = (2, 276),
        noise_shape: Tuple[int, int] = (1, 128),
        cond_mask_prob: float = 0.0,
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        self.h_dim = h_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.history_shape = tuple(int(x) for x in history_shape)
        self.noise_shape = tuple(int(x) for x in noise_shape)
        self.clip_dim = clip_dim
        self.cond_mask_prob = float(cond_mask_prob)

        self.sequence_pos_encoder = PositionalEncoding(self.h_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(
            self.h_dim,
            self.sequence_pos_encoder,
            rngs=rngs,
        )
        self.embed_text = TorchLinear(self.clip_dim, self.h_dim, rngs=rngs)
        self.embed_history = TorchLinear(self.history_shape[-1], self.h_dim, rngs=rngs)
        self.embed_noise = TorchLinear(self.noise_shape[-1], self.h_dim, rngs=rngs)
        self.seqTransEncoder = TransformerEncoder(
            self.h_dim,
            self.num_heads,
            self.ff_size,
            self.dropout,
            self.activation,
            self.num_layers,
            rngs=rngs,
        )
        self.output_process = TorchLinear(self.h_dim, self.noise_shape[-1], rngs=rngs)

    def __call__(
        self,
        x_t,
        timesteps,
        y,
        *,
        training: bool = False,
        rng: Optional[jax.Array] = None,
    ):
        timesteps = _require_integer_timesteps(timesteps, "DenoiserTransformer")
        emb_time = self.embed_timestep(timesteps)
        emb_history = jnp.swapaxes(
            self.embed_history(y["history_motion_normalized"]),
            0,
            1,
        )
        emb_text = self.embed_text(
            self.mask_cond(
                y["text_embedding"],
                force_mask=bool(y.get("uncond", False)),
                training=training,
                rng=rng,
            )
        )[None, :, :]
        emb_noise = jnp.swapaxes(self.embed_noise(x_t), 0, 1)
        xseq = jnp.concatenate((emb_time, emb_text, emb_history, emb_noise), axis=0)
        xseq = self.sequence_pos_encoder(xseq)
        output = self.seqTransEncoder(xseq)[-self.noise_shape[0] :]
        output = self.output_process(output)
        return jnp.swapaxes(output, 0, 1)


def _parse_shape(value: str) -> Tuple[int, ...]:
    parts = [int(part) for part in value.replace(",", " ").split()]
    if not parts:
        raise argparse.ArgumentTypeError("shape must contain at least one integer")
    return tuple(parts)


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test the JAX DART denoisers.")
    parser.add_argument("--model-type", default="mlp", choices=["mlp", "transformer"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--h-dim", type=int, default=512)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--ff-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", default="gelu")
    parser.add_argument("--clip-dim", type=int, default=512)
    parser.add_argument("--history-shape", type=_parse_shape, default=(2, 276))
    parser.add_argument("--noise-shape", type=_parse_shape, default=(1, 128))
    return parser.parse_args()


def main():
    args = parse_args()
    from flax import nnx

    rng = jax.random.PRNGKey(args.seed)
    if args.model_type == "transformer":
        model = DenoiserTransformer(
            h_dim=args.h_dim,
            ff_size=args.ff_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            activation=args.activation,
            clip_dim=args.clip_dim,
            history_shape=args.history_shape,
            noise_shape=args.noise_shape,
            rngs=nnx.Rngs(args.seed),
        )
    else:
        model = DenoiserMLP(
            h_dim=args.h_dim,
            n_blocks=args.n_blocks,
            dropout=args.dropout,
            activation=args.activation,
            clip_dim=args.clip_dim,
            history_shape=args.history_shape,
            noise_shape=args.noise_shape,
            rngs=nnx.Rngs(args.seed),
        )
    data_keys = jax.random.split(rng, 3)
    x_t = jax.random.normal(data_keys[0], (args.batch_size, *args.noise_shape))
    history = jax.random.normal(data_keys[1], (args.batch_size, *args.history_shape))
    text = jax.random.normal(data_keys[2], (args.batch_size, args.clip_dim))
    timesteps = jnp.arange(args.batch_size, dtype=jnp.int32)
    output = model(
        x_t,
        timesteps,
        {
            "history_motion_normalized": history,
            "text_embedding": text,
        },
    )
    print("backend:", jax.default_backend())
    print("x_t:", x_t.shape, x_t.dtype)
    print("history:", history.shape, history.dtype)
    print("text:", text.shape, text.dtype)
    print("output:", output.shape, output.dtype)


if __name__ == "__main__":
    main()
