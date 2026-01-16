"""LVAE core components: autoencoder architectures, pruning, constraints, and shared layers.

This module provides the dimension-agnostic building blocks for Least Volume Autoencoder (LVAE)
implementations. It consolidates shared functionality used across both 1D and 2D LVAE models.

Modules:
    aes: Autoencoder architectures with volume regularization and dynamic pruning
    constraint_handlers: Constrained optimization methods for multi-objective training
    components: Shared neural network components, utilities, and schedules

Example:
    >>> from engiopt.lvae_core import (
    ...     LeastVolumeAE_DynamicPruning,
    ...     polynomial_schedule,
    ...     create_constraint_handler,
    ... )
"""

from .aes import (
    AutoEncoder,
    ConstrainedDesignLeastVolumeAE_DP,
    DesignLeastVolumeAE_DP,
    InterpretableDesignLeastVolumeAE_DP,
    LeastVolumeAE,
    LeastVolumeAE_DynamicPruning,
    PruningPolicy,
    VAE,
)
from .components import (
    MLP,
    SNMLP,
    SNLinearCombo,
    Normalizer,
    Scale,
    TrueSNDeconv2DCombo,
    polynomial_schedule,
    spectral_norm_conv,
)
from .constraint_handlers import (
    ConstraintHandler,
    ConstraintLosses,
    ConstraintThresholds,
    create_constraint_handler,
)

__all__ = [
    # Autoencoder models
    "AutoEncoder",
    "VAE",
    "LeastVolumeAE",
    "LeastVolumeAE_DynamicPruning",
    "DesignLeastVolumeAE_DP",
    "InterpretableDesignLeastVolumeAE_DP",
    "ConstrainedDesignLeastVolumeAE_DP",
    "PruningPolicy",
    # Constraint handling
    "ConstraintHandler",
    "ConstraintLosses",
    "ConstraintThresholds",
    "create_constraint_handler",
    # Utilities and components
    "polynomial_schedule",
    "spectral_norm_conv",
    "Scale",
    "Normalizer",
    "MLP",
    "SNMLP",
    "SNLinearCombo",
    "TrueSNDeconv2DCombo",
]
