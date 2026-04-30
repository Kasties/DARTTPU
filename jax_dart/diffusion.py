"""JAX diffusion helpers matching DART's Torch ``q_sample`` path."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


def get_named_beta_schedule(
    schedule_name: str,
    num_diffusion_timesteps: int,
    scale_betas: float = 1.0,
) -> np.ndarray:
    """Return the same beta schedules as ``diffusion/gaussian_diffusion.py``."""
    if schedule_name == "linear":
        scale = scale_betas * 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start,
            beta_end,
            num_diffusion_timesteps,
            dtype=np.float64,
        )
    if schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(
    num_diffusion_timesteps: int,
    alpha_bar,
    max_beta: float = 0.999,
) -> np.ndarray:
    """Discretize a cumulative alpha-bar function into betas."""
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.asarray(betas, dtype=np.float64)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    import jax.numpy as jnp

    values = jnp.asarray(arr, dtype=jnp.float32)[timesteps]
    while len(values.shape) < len(broadcast_shape):
        values = values[..., None]
    return jnp.broadcast_to(values, broadcast_shape)


def q_sample(
    x_start,
    t,
    *,
    sqrt_alphas_cumprod,
    sqrt_one_minus_alphas_cumprod,
    noise=None,
    rng=None,
):
    """Sample ``q(x_t | x_0)`` with Torch-compatible float32 extraction."""
    import jax
    import jax.numpy as jnp

    if noise is None:
        if rng is None:
            raise ValueError("q_sample requires either explicit noise or a JAX rng.")
        noise = jax.random.normal(rng, x_start.shape, dtype=x_start.dtype)
    noise = jnp.asarray(noise, dtype=x_start.dtype)
    if noise.shape != x_start.shape:
        raise ValueError(f"noise shape {noise.shape} does not match x_start {x_start.shape}")
    return (
        _extract_into_tensor(sqrt_alphas_cumprod, t, x_start.shape) * x_start
        + _extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
    )


@dataclass(frozen=True)
class GaussianDiffusion:
    """Minimal JAX diffusion state for forward noising parity."""

    betas: np.ndarray
    sqrt_alphas_cumprod: np.ndarray
    sqrt_one_minus_alphas_cumprod: np.ndarray
    rescale_timesteps: bool = False

    @classmethod
    def create(
        cls,
        *,
        diffusion_steps: int = 10,
        noise_schedule: str = "cosine",
        scale_betas: float = 1.0,
        rescale_timesteps: bool = False,
    ) -> "GaussianDiffusion":
        betas = get_named_beta_schedule(noise_schedule, diffusion_steps, scale_betas)
        return cls.from_betas(betas, rescale_timesteps=rescale_timesteps)

    @classmethod
    def from_betas(
        cls,
        betas,
        *,
        rescale_timesteps: bool = False,
    ) -> "GaussianDiffusion":
        betas = np.asarray(betas, dtype=np.float64)
        if betas.ndim != 1:
            raise ValueError("betas must be 1-D")
        if not np.all((betas > 0) & (betas <= 1)):
            raise ValueError("betas must be in (0, 1]")
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        return cls(
            betas=betas,
            sqrt_alphas_cumprod=np.sqrt(alphas_cumprod),
            sqrt_one_minus_alphas_cumprod=np.sqrt(1.0 - alphas_cumprod),
            rescale_timesteps=rescale_timesteps,
        )

    @property
    def num_timesteps(self) -> int:
        return int(self.betas.shape[0])

    def scale_timesteps(self, t):
        import jax.numpy as jnp

        if self.rescale_timesteps:
            return t.astype(jnp.float32) * (1000.0 / self.num_timesteps)
        return t

    def q_sample(self, x_start, t, noise=None, rng=None):
        return q_sample(
            x_start,
            t,
            sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
            noise=noise,
            rng=rng,
        )


def create_gaussian_diffusion(
    *,
    diffusion_steps: int = 10,
    noise_schedule: str = "cosine",
    scale_betas: float = 1.0,
    rescale_timesteps: bool = False,
) -> GaussianDiffusion:
    return GaussianDiffusion.create(
        diffusion_steps=diffusion_steps,
        noise_schedule=noise_schedule,
        scale_betas=scale_betas,
        rescale_timesteps=rescale_timesteps,
    )
