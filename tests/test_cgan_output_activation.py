"""Focused checks for configurable cGAN generator output activations."""

import pytest


@pytest.mark.parametrize(
    ("module_name", "latent_shape", "condition_shape", "design_shape", "activation_getter"),
    [
        (
            "engiopt.cgan_2d.cgan_2d",
            (2, 4),
            (2, 1),
            (3, 3),
            lambda model: model.model[-1],
        ),
        (
            "engiopt.cgan_cnn_2d.cgan_cnn_2d",
            (2, 4, 1, 1),
            (2, 1, 1, 1),
            (8, 8),
            lambda model: model.up_blocks[-1],
        ),
    ],
)
def test_cgan_generators_keep_tanh_as_default(
    module_name: str,
    latent_shape: tuple[int, ...],
    condition_shape: tuple[int, ...],
    design_shape: tuple[int, ...],
    activation_getter,
) -> None:
    """Existing checkpoints remain interpretable because tanh is still the default."""
    th = pytest.importorskip("torch")
    cgan_module = pytest.importorskip(module_name)

    generator = cgan_module.Generator(latent_dim=4, n_conds=1, design_shape=design_shape)

    assert isinstance(activation_getter(generator), th.nn.Tanh)


@pytest.mark.parametrize(
    ("module_name", "latent_shape", "condition_shape", "design_shape", "activation_getter"),
    [
        (
            "engiopt.cgan_2d.cgan_2d",
            (2, 4),
            (2, 1),
            (3, 3),
            lambda model: model.model[-1],
        ),
        (
            "engiopt.cgan_cnn_2d.cgan_cnn_2d",
            (2, 4, 1, 1),
            (2, 1, 1, 1),
            (8, 8),
            lambda model: model.up_blocks[-1],
        ),
    ],
)
def test_cgan_generators_can_emit_density_range_with_sigmoid(
    module_name: str,
    latent_shape: tuple[int, ...],
    condition_shape: tuple[int, ...],
    design_shape: tuple[int, ...],
    activation_getter,
) -> None:
    """New density-field runs can opt into native [0, 1] generator outputs."""
    th = pytest.importorskip("torch")
    cgan_module = pytest.importorskip(module_name)

    generator = cgan_module.Generator(
        latent_dim=4,
        n_conds=1,
        design_shape=design_shape,
        generator_output_activation="sigmoid",
    )
    generator.eval()

    with th.no_grad():
        generated = generator(th.randn(latent_shape), th.randn(condition_shape))

    assert isinstance(activation_getter(generator), th.nn.Sigmoid)
    assert th.all(generated >= 0)
    assert th.all(generated <= 1)
