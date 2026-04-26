"""Flax NNX port of DART's ``model/mld_vae.py``.

This first port is built for parity testing:

* tensors are batch-first at the public method boundary;
* latent tensors match Torch as ``[latent_size, batch, latent_dim]``;
* core layer parameter layouts mirror PyTorch where practical.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from jax_dart.data.primitive_shards import iter_vae_batches, load_metadata

from .mld_transformer import (
    SkipTransformerDecoder,
    SkipTransformerEncoder,
    build_position_encoding,
)
from .torch_compatible import TorchLinear


@dataclass(frozen=True)
class NormalParams:
    loc: jax.Array
    scale: jax.Array
    logvar: jax.Array


def kl_standard_normal(dist: NormalParams):
    return 0.5 * jnp.mean(dist.loc**2 + dist.scale**2 - 1.0 - dist.logvar)


def huber_loss(pred, target, delta: float = 1.0):
    error = pred - target
    abs_error = jnp.abs(error)
    quadratic = jnp.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return 0.5 * quadratic**2 + delta * linear


def vae_reconstruction_kl_loss(
    future_pred,
    future_gt,
    dist: NormalParams,
    weight_rec: float = 1.0,
    weight_kl: float = 1e-4,
):
    rec = jnp.mean(huber_loss(future_pred, future_gt))
    kl = kl_standard_normal(dist)
    return weight_rec * rec + weight_kl * kl, {"rec": rec, "kl": kl}


class AutoMldVae(nnx.Module):
    def __init__(
        self,
        nfeats: int,
        latent_dim: tuple[int, int] = (1, 256),
        h_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 7,
        num_heads: int = 4,
        dropout: float = 0.1,
        arch: str = "all_encoder",
        normalize_before: bool = False,
        activation: str = "gelu",
        position_embedding: str = "learned",
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1]
        self.h_dim = h_dim
        self.arch = arch
        self.nfeats = nfeats

        self.query_pos_encoder = build_position_encoding(
            h_dim, position_embedding=position_embedding, rngs=rngs
        )
        self.query_pos_decoder = build_position_encoding(
            h_dim, position_embedding=position_embedding, rngs=rngs
        )
        self.encoder = SkipTransformerEncoder(
            h_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
            num_layers,
            rngs=rngs,
        )
        self.encoder_latent_proj = TorchLinear(h_dim, self.latent_dim, rngs=rngs)

        if arch == "all_encoder":
            self.decoder = SkipTransformerEncoder(
                h_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
                num_layers,
                rngs=rngs,
            )
        elif arch == "encoder_decoder":
            self.decoder = SkipTransformerDecoder(
                h_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
                num_layers,
                rngs=rngs,
            )
        else:
            raise ValueError(f"Unsupported VAE architecture {arch}.")

        self.decoder_latent_proj = TorchLinear(self.latent_dim, h_dim, rngs=rngs)
        self.global_motion_token = nnx.Param(
            jax.random.normal(rngs.params(), (self.latent_size * 2, h_dim))
        )
        self.skel_embedding = TorchLinear(nfeats, h_dim, rngs=rngs)
        self.final_layer = TorchLinear(h_dim, nfeats, rngs=rngs)
        self.latent_mean = nnx.Variable(jnp.array(0.0, dtype=jnp.float32))
        self.latent_std = nnx.Variable(jnp.array(1.0, dtype=jnp.float32))

    def encode(self, future_motion, history_motion, rng=None, scale_latent: bool = False):
        batch_size = future_motion.shape[0]
        x = jnp.concatenate((history_motion, future_motion), axis=1)
        x = self.skel_embedding(x)
        dist_tokens = jnp.broadcast_to(
            self.global_motion_token.value[None, :, :],
            (batch_size, self.latent_size * 2, self.h_dim),
        )
        xseq = jnp.concatenate((dist_tokens, x), axis=1)
        xseq = self.query_pos_encoder(xseq)
        dist = self.encoder(xseq)[:, : dist_tokens.shape[1], :]
        dist = self.encoder_latent_proj(dist)
        dist = jnp.swapaxes(dist, 0, 1)
        mu = dist[: self.latent_size]
        logvar = jnp.clip(dist[self.latent_size :], min=-10.0, max=10.0)
        std = jnp.exp(0.5 * logvar)
        if rng is None:
            latent = mu
        else:
            latent = mu + std * jax.random.normal(rng, mu.shape)
        if scale_latent:
            latent = latent / self.latent_std.value
        return latent, NormalParams(loc=mu, scale=std, logvar=logvar)

    def decode(self, z, history_motion, nfuture: int, scale_latent: bool = False):
        batch_size = history_motion.shape[0]
        if scale_latent:
            z = z * self.latent_std.value
        z = self.decoder_latent_proj(z)
        z = jnp.swapaxes(z, 0, 1)
        queries = jnp.zeros((batch_size, nfuture, self.h_dim), dtype=history_motion.dtype)
        history_embedding = self.skel_embedding(history_motion)

        if self.arch == "all_encoder":
            xseq = jnp.concatenate((z, history_embedding, queries), axis=1)
            xseq = self.query_pos_decoder(xseq)
            output = self.decoder(xseq)[:, -nfuture:, :]
        else:
            xseq = jnp.concatenate((history_embedding, queries), axis=1)
            xseq = self.query_pos_decoder(xseq)
            output = self.decoder(xseq, z)[:, -nfuture:, :]

        return self.final_layer(output)


def _parse_latent_dim(value: str):
    parts = [int(part) for part in value.replace(",", " ").split()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("latent dim must contain two integers, e.g. '1 256'")
    return tuple(parts)


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test the JAX NNX DART VAE.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--batch-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--h-dim", type=int, default=256)
    parser.add_argument("--ff-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=7)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--latent-dim", type=_parse_latent_dim, default=(1, 256))
    parser.add_argument("--arch", default="all_encoder", choices=["all_encoder", "encoder_decoder"])
    return parser.parse_args()


def _jax_dtype(name: str):
    if name == "bf16":
        return jnp.bfloat16
    if name == "fp16":
        return jnp.float16
    if name == "fp32":
        return jnp.float32
    raise ValueError(f"Unsupported dtype {name}.")


def main():
    args = parse_args()
    metadata = load_metadata(args.root)
    feature_dim = int(metadata["primitive"]["feature_dim"])
    batch = next(iter_vae_batches(args.root, batch_samples=args.batch_samples))
    dtype = _jax_dtype(args.dtype)
    history = jnp.asarray(batch["history"], dtype=dtype)
    future = jnp.asarray(batch["future"], dtype=dtype)

    model = AutoMldVae(
        nfeats=feature_dim,
        latent_dim=args.latent_dim,
        h_dim=args.h_dim,
        ff_size=args.ff_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        arch=args.arch,
        rngs=nnx.Rngs(args.seed),
    )

    @nnx.jit
    def forward_smoke(model, history, future):
        latent, dist = model.encode(future, history, rng=None)
        future_pred = model.decode(latent, history, nfuture=future.shape[1])
        loss, terms = vae_reconstruction_kl_loss(future_pred, future, dist)
        return latent, future_pred, loss, terms["rec"], terms["kl"]

    latent, future_pred, loss, rec, kl = forward_smoke(model, history, future)
    print("backend:", jax.default_backend())
    print("devices:", jax.devices())
    print("history:", history.shape, history.dtype)
    print("future:", future.shape, future.dtype)
    print("latent:", latent.shape, latent.dtype)
    print("future_pred:", future_pred.shape, future_pred.dtype)
    print("loss:", loss)
    print("rec:", rec)
    print("kl:", kl)


if __name__ == "__main__":
    main()

