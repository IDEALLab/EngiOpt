"""
Sampling strategies for the LAE (legacy module, kept for checkpoint compatibility).

The model is deterministic given (c, x0): decoding with z=0 yields physically
valid wings (confirmed by the z=0 diagnostic: cd=5.15, cl=0.80 at the target
conditions).  Generation varies the inputs (c and x0), not z.
This module retains only the LAESampler stub required to unpickle existing checkpoints.
"""

import torch


class LAESampler:
    """Stub retained for checkpoint backward compatibility.

    Generation uses z=0 with varied (c, x0) inputs.
    """

    def __init__(self, latent_dim: int = 64, **kwargs):
        self.latent_dim = latent_dim


# Backward-compat alias
LVAESampler = LAESampler
