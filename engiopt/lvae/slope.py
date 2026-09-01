"""Measure what a decoder's Lipschitz constant actually is.

Least Volume's pruning guarantee (Chen & Fuge, Theorem 11) says that freezing a
set of latent dimensions raises reconstruction error by at most
`K * sqrt(sum of their variances)`. Every statement the metric suite makes about
intrinsic dimension inherits that bound, so `K` has to be real.

Two numbers say whether it is, and they are different questions:

- `certified_slope` multiplies the operator norms of the decoder's linear stages.
  This is the bound the architecture enforces. It is only equal to
  `lipschitz_scale` if every stage really is normalised to 1.
- `measured_slope` is the largest stretch actually observed between random latent
  pairs. It is a lower bound on the true constant, and it cannot exceed the
  certified one unless the certification is wrong.

`certified_slope` uses power iteration against each layer as a function, which
is the same thing Chen recommends for convolutions and is why it disagrees with
`torch.nn.utils.parametrizations.spectral_norm`: that normalises the largest
singular value of the weight's `(out_channels, -1)` reshape, which for a
convolution is a lower bound on the operator norm rather than the operator norm.

Run it on the pinned instruments after any architecture change::

    python -m engiopt.lvae.slope --problem-id photonics2d
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch as th
from torch import nn

if TYPE_CHECKING:
    from engiopt.lvae.checkpoints import LoadedLVAE

# Layers that are linear maps, so a singular value is defined for them. LeakyReLU
# and the like are 1-Lipschitz and contribute a factor of 1, so they are skipped.
LINEAR_LAYERS = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)


def operator_norm(layer: nn.Module, input_shape: tuple[int, ...], n_iter: int = 100, seed: int = 0) -> float:
    """Largest singular value of a linear layer, by power iteration on the layer itself.

    Applies `A` by calling the layer and `A^T` by differentiating through it, so
    it needs no per-layer adjoint bookkeeping and is exact for convolutions of
    any stride, padding or dilation. Bias is a translation and does not affect
    the constant, so it is left in place and cancels in the difference.

    Args:
        layer: A linear layer.
        input_shape: Shape of one input to `layer`, without the batch dimension.
        n_iter: Power iterations. The ratio settles well before 100 here.
        seed: Seed for the starting vector.

    Returns:
        The operator norm, i.e. `max ||layer(u) - layer(0)|| / ||u||`.
    """
    generator = th.Generator().manual_seed(seed)
    u = th.randn(1, *input_shape, generator=generator)
    u = u / u.norm()

    # Subtracting layer(0) removes the bias, leaving the linear part.
    with th.no_grad():
        offset = layer(th.zeros(1, *input_shape))

    for _ in range(n_iter):
        u = u.detach().requires_grad_(requires_grad=True)
        v = layer(u) - offset
        # grad of <v, v'> wrt u with v' held fixed is A^T v, so this is A^T A u.
        (adjoint,) = th.autograd.grad(v, u, grad_outputs=v)
        u = (adjoint / adjoint.norm()).detach()

    with th.no_grad():
        return float((layer(u) - offset).norm() / u.norm())


def layer_norms(decoder: nn.Module, latent_dim: int) -> list[tuple[str, float]]:
    """Operator norm of every linear stage in a decoder, in forward order.

    Input shapes are recorded from a real forward pass rather than derived by
    hand, so this keeps working when the architecture changes.

    Args:
        decoder: The decoder to walk.
        latent_dim: Width of the latent the decoder consumes.

    Returns:
        `(name, operator_norm)` per linear stage.
    """
    shapes: dict[str, tuple[int, ...]] = {}
    handles = []

    def record(name: str):
        def hook(_module: nn.Module, inputs: tuple[th.Tensor, ...]) -> None:
            shapes[name] = tuple(inputs[0].shape[1:])

        return hook

    for name, module in decoder.named_modules():
        if isinstance(module, LINEAR_LAYERS):
            handles.append(module.register_forward_pre_hook(record(name)))

    with th.no_grad():
        decoder(th.zeros(1, latent_dim))
    for handle in handles:
        handle.remove()

    return [
        (name, operator_norm(module, shapes[name]))
        for name, module in decoder.named_modules()
        if isinstance(module, LINEAR_LAYERS)
    ]


def certified_slope(decoder: nn.Module, latent_dim: int) -> float:
    """The bound the architecture enforces: the product of its layer norms.

    Includes `lipschitz_scale`, which the decoder applies to its output, so this
    is directly comparable to the configured cap.

    Args:
        decoder: The decoder to certify.
        latent_dim: Width of the latent the decoder consumes.

    Returns:
        The product of every linear stage's operator norm, times the output scale.
    """
    product = 1.0
    for _, norm in layer_norms(decoder, latent_dim):
        product *= norm
    return product * float(getattr(decoder, "lipschitz_scale", 1.0))


def measured_slope(decoder: nn.Module, latent_dim: int, n_pairs: int = 512, seed: int = 0) -> float:
    """Largest stretch observed between random latent pairs.

    A lower bound on the true Lipschitz constant. If it exceeds `certified_slope`
    then the certification is wrong, which is the check worth running.

    Args:
        decoder: The decoder to probe.
        latent_dim: Width of the latent the decoder consumes.
        n_pairs: Latent pairs to try.
        seed: Seed for the pairs.

    Returns:
        `max ||dec(z1) - dec(z2)|| / ||z1 - z2||` over the sampled pairs.
    """
    generator = th.Generator().manual_seed(seed)
    z1 = th.randn(n_pairs, latent_dim, generator=generator)
    z2 = th.randn(n_pairs, latent_dim, generator=generator)

    with th.no_grad():
        dx = (decoder(z1) - decoder(z2)).flatten(1).norm(dim=1)
    dz = (z1 - z2).norm(dim=1)
    return float((dx / dz).max())


def report(lvae: LoadedLVAE) -> dict[str, float]:
    """Certified and measured slope for a loaded instrument, with its configured cap.

    Args:
        lvae: A loaded LVAE whose decoder can be built.

    Returns:
        `configured`, `certified` and `measured` slopes. `certified / configured`
        is the factor by which the bound is violated, and is 1.0 when it holds.
    """
    decoder = lvae.decoder
    latent_dim = lvae.config.latent_dim
    return {
        "configured": float(lvae.config.decoder_lipschitz_scale),
        "certified": certified_slope(decoder, latent_dim),
        "measured": measured_slope(decoder, latent_dim),
    }
