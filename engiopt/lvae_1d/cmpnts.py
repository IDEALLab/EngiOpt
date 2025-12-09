"""Neural network components for 1D LVAE models with Lipschitz constraints.

This module provides building blocks for constructing spectrally normalized
encoders and decoders that maintain 1-Lipschitz bounds, crucial for stable
training of Least Volume Autoencoders (LVAE).

Key Components:
- Normalizer: Min-max normalization/denormalization utility
- Scale: Learnable or fixed scaling layer
- MLP: Multi-layer perceptron with optional spectral normalization
- SNMLP: Spectrally normalized MLP (Lipschitz ≤ 1)
- Conv1DEncoder: 1D convolutional encoder
- SNConv1DEncoder: Spectrally normalized 1D convolutional encoder
- BezierLayer: Rational Bezier curve generation layer
- SNBezierDecoder: Spectrally normalized Bezier-based decoder for airfoils
- FactorizedConv1DEncoder: Factorized encoder for sequences + auxiliary scalars
- FactorizedBezierDecoder: Factorized Bezier decoder for curves + auxiliary outputs

References:
- Spectral Normalization: https://arxiv.org/abs/1802.05957
- Least Volume Autoencoder: (add paper reference)
"""

from __future__ import annotations

import torch as th
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


class Normalizer:
    """Min-max normalization utility for tensors.

    Provides normalization and denormalization using min-max scaling to a
    configurable target range. Useful for preprocessing inputs and postprocessing
    outputs in neural networks.

    Args:
        min_val: Minimum value(s) for normalization (scalar or tensor)
        max_val: Maximum value(s) for normalization (scalar or tensor)
        target_range: Target range for normalized values (default: [0, 1])
        eps: Small constant to avoid division by zero (default: 1e-7)

    Example:
        >>> # Normalize angles from [-10, 10] to [0, 1]
        >>> normalizer = Normalizer(min_val=-10.0, max_val=10.0)
        >>> angle_norm = normalizer.normalize(angle)  # → [0, 1]
        >>> angle_orig = normalizer.denormalize(angle_norm)  # → [-10, 10]
    """

    def __init__(
        self,
        min_val: float | th.Tensor,
        max_val: float | th.Tensor,
        target_range: tuple[float, float] = (0.0, 1.0),
        eps: float = 1e-7,
    ):
        """Initialize normalizer with min and max values.

        Args:
            min_val: Minimum value(s) for normalization
            max_val: Maximum value(s) for normalization
            target_range: Target range for normalization (default: [0, 1])
            eps: Small constant to avoid division by zero
        """
        self.eps = eps
        self.min_val = th.as_tensor(min_val) if not isinstance(min_val, th.Tensor) else min_val
        self.max_val = th.as_tensor(max_val) if not isinstance(max_val, th.Tensor) else max_val
        self.target_min, self.target_max = target_range

    def normalize(self, x: th.Tensor) -> th.Tensor:
        """Normalize input to target range.

        Args:
            x: Input tensor in original range

        Returns:
            Normalized tensor in target range
        """
        # First normalize to [0, 1]
        x_01 = (x - self.min_val) / (self.max_val - self.min_val + self.eps)
        # Then scale to target range
        return x_01 * (self.target_max - self.target_min) + self.target_min

    def denormalize(self, x: th.Tensor) -> th.Tensor:
        """Denormalize input from target range to original range.

        Args:
            x: Normalized tensor in target range

        Returns:
            Denormalized tensor in original range
        """
        # First scale from target range to [0, 1]
        x_01 = (x - self.target_min) / (self.target_max - self.target_min + self.eps)
        # Then denormalize to original range
        return x_01 * (self.max_val - self.min_val + self.eps) + self.min_val


class Scale(nn.Module):
    """Learnable or fixed scalar multiplication layer.

    This layer multiplies input by a scalar value, useful for controlling
    the Lipschitz constant of the overall network.

    Args:
        scale: Initial scale value (can be float or tensor)

    Shape:
        - Input: (*, H_in) where * means any number of dimensions
        - Output: (*, H_in) same shape as input
    """

    def __init__(self, scale: float | th.Tensor) -> None:
        super().__init__()
        self.register_buffer("s", th.as_tensor(scale, dtype=th.float32))

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Multiply input by scale factor.

        Args:
            x: Input tensor

        Returns:
            Scaled tensor: x * scale
        """
        return self.s * x


class MLP(nn.Module):
    """Multi-layer perceptron with flexible architecture.

    Constructs a fully connected network with specified hidden layer widths.
    The final layer is always a linear transformation without activation.
    Hidden layers use LeakyReLU(0.2) activation by default.

    Args:
        in_features: Number of input features
        out_features: Number of output features
        hidden_widths: List of hidden layer widths (empty list = single linear layer)
        activation: Activation function for hidden layers (default: LeakyReLU(0.2))
        use_spectral_norm: Whether to apply spectral normalization (default: False)

    Shape:
        - Input: (N, H_in) where H_in = in_features
        - Output: (N, H_out) where H_out = out_features

    Example:
        >>> mlp = MLP(128, 64, hidden_widths=[256, 128])
        >>> # Network: 128 -> 256 -> 128 -> 64
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_widths: list[int],
        activation: nn.Module | None = None,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_widths = list(hidden_widths)

        if activation is None:
            activation = nn.LeakyReLU(0.2)

        # Build layer specifications: [in_features] + hidden_widths + [out_features]
        layer_sizes = (
            [in_features] + self.hidden_widths + [out_features]
        )

        # Build sequential model
        layers: list[nn.Module] = []
        for i in range(len(layer_sizes) - 1):
            # Create linear layer
            linear = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            if use_spectral_norm:
                linear = spectral_norm(linear)
            layers.append(linear)

            # Add activation for all layers except the last
            if i < len(layer_sizes) - 2:
                layers.append(activation)

        self.model = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward pass through MLP.

        Args:
            x: Input tensor (N, in_features)

        Returns:
            Output tensor (N, out_features)
        """
        return self.model(x)


class SNMLP(MLP):
    """Spectrally normalized MLP with Lipschitz constant ≤ 1.

    This is a convenience wrapper around MLP that automatically enables
    spectral normalization on all linear layers.

    Args:
        in_features: Number of input features
        out_features: Number of output features
        hidden_widths: List of hidden layer widths
        activation: Activation function for hidden layers (default: LeakyReLU(0.2))

    Shape:
        - Input: (N, H_in) where H_in = in_features
        - Output: (N, H_out) where H_out = out_features
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_widths: list[int],
        activation: nn.Module | None = None,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_widths=hidden_widths,
            activation=activation,
            use_spectral_norm=True,
        )


class Conv1DEncoder(nn.Module):
    """1D Convolutional encoder for sequences.

    Encodes 1D sequences (e.g., airfoil coordinates) into a latent representation
    using strided convolutions followed by an MLP. Architecture inspired by DCGAN.

    Args:
        in_channels: Number of input channels (e.g., 2 for x,y coordinates)
        in_features: Length of input sequence
        latent_dim: Dimensionality of latent code
        conv_channels: List of channel sizes for conv layers (e.g., [64, 128, 256])
        mlp_hidden: List of hidden layer widths for final MLP
        use_spectral_norm: Whether to apply spectral normalization (default: False)

    Shape:
        - Input: (N, C, L) where C = in_channels, L = in_features
        - Output: (N, latent_dim)

    Example:
        >>> encoder = Conv1DEncoder(
        ...     in_channels=2,
        ...     in_features=192,
        ...     latent_dim=100,
        ...     conv_channels=[64, 128, 256],
        ...     mlp_hidden=[512, 256]
        ... )
        >>> z = encoder(coords)  # coords shape: (B, 2, 192)
    """

    def __init__(
        self,
        in_channels: int,
        in_features: int,
        latent_dim: int,
        conv_channels: list[int],
        mlp_hidden: list[int],
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.in_features = in_features
        self.latent_dim = latent_dim
        self.conv_channels = conv_channels

        # Build convolutional layers
        conv_layers: list[nn.Module] = []
        channels_sequence = [in_channels] + conv_channels

        for i in range(len(channels_sequence) - 1):
            # Conv: kernel=4, stride=2, padding=1 (halves spatial size)
            conv = nn.Conv1d(
                channels_sequence[i],
                channels_sequence[i + 1],
                kernel_size=4,
                stride=2,
                padding=1,
            )
            if use_spectral_norm:
                conv = spectral_norm(conv)

            conv_layers.append(conv)
            conv_layers.append(nn.LeakyReLU(0.2))

        self.conv = nn.Sequential(*conv_layers)

        # Calculate flattened size after convolutions
        # Each conv layer with stride=2 halves the spatial dimension
        n_conv_layers = len(conv_channels)
        final_length = in_features // (2 ** n_conv_layers)
        m_features = final_length * conv_channels[-1]

        # MLP to project to latent space
        self.mlp = MLP(
            in_features=m_features,
            out_features=latent_dim,
            hidden_widths=mlp_hidden,
            use_spectral_norm=use_spectral_norm,
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Encode sequence to latent code.

        Args:
            x: Input sequence (N, in_channels, in_features)

        Returns:
            Latent code (N, latent_dim)
        """
        h = self.conv(x)
        h = h.flatten(1)  # Flatten spatial dimensions
        z = self.mlp(h)
        return z


class SNConv1DEncoder(Conv1DEncoder):
    """Spectrally normalized 1D convolutional encoder.

    This is a convenience wrapper that automatically enables spectral normalization
    on all convolutional and linear layers to maintain Lipschitz constant ≤ 1.

    Args:
        in_channels: Number of input channels
        in_features: Length of input sequence
        latent_dim: Dimensionality of latent code
        conv_channels: List of channel sizes for conv layers
        mlp_hidden: List of hidden layer widths for final MLP

    Shape:
        - Input: (N, C, L) where C = in_channels, L = in_features
        - Output: (N, latent_dim)
    """

    def __init__(
        self,
        in_channels: int,
        in_features: int,
        latent_dim: int,
        conv_channels: list[int],
        mlp_hidden: list[int],
    ):
        super().__init__(
            in_channels=in_channels,
            in_features=in_features,
            latent_dim=latent_dim,
            conv_channels=conv_channels,
            mlp_hidden=mlp_hidden,
            use_spectral_norm=True,
        )


class BezierLayer(nn.Module):
    """Rational Bezier curve generation layer.

    Generates Bezier curves using rational Bezier basis functions. Given control
    points, weights, and features for determining parameter values, computes a
    sequence of data points along the curve.

    The rational Bezier curve is defined as:
        C(t) = sum(w_i * P_i * B_i(t)) / sum(w_i * B_i(t))

    where B_i(t) are Bernstein basis polynomials.

    Args:
        m_features: Number of input features for generating intervals
        n_control_points: Number of control points
        n_data_points: Number of output data points along the curve
        eps: Small constant for numerical stability (default: 1e-7)

    Shape:
        - Input features: (N, m_features)
        - Input control_points: (N, 2, n_control_points)
        - Input weights: (N, 1, n_control_points)
        - Output data_points: (N, 2, n_data_points)
        - Output param_values: (N, 1, n_data_points)
        - Output intervals: (N, n_data_points)

    Note:
        This layer is NOT spectrally normalized by default. The interval generation
        uses a linear layer that should be spectrally normalized if Lipschitz bounds
        are required.
    """

    def __init__(
        self,
        m_features: int,
        n_control_points: int,
        n_data_points: int,
        eps: float = 1e-7,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.n_control_points = n_control_points
        self.n_data_points = n_data_points
        self.eps = eps

        # Generate parameter values t ∈ [0,1] from features
        # The network learns to distribute points along the curve
        interval_gen = nn.Linear(m_features, n_data_points - 1)
        if use_spectral_norm:
            interval_gen = spectral_norm(interval_gen)

        self.generate_intervals = nn.Sequential(
            interval_gen,
            nn.Softmax(dim=1),  # Softmax: Lipschitz constant ≤ 1
            nn.ConstantPad1d((1, 0), 0.0),  # Add leading zero: Lipschitz = 1
        )

    def forward(
        self,
        features: th.Tensor,
        control_points: th.Tensor,
        weights: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Generate Bezier curve from control points and weights.

        Args:
            features: Input features for interval generation (N, m_features)
            control_points: Control points (N, 2, n_control_points)
            weights: Weights for rational Bezier (N, 1, n_control_points)

        Returns:
            Tuple of:
                - data_points: Points along curve (N, 2, n_data_points)
                - param_values: Parameter values used (N, 1, n_data_points)
                - intervals: Interval sizes (N, n_data_points)
        """
        # Generate parameter values t in [0, 1]
        intervals = self.generate_intervals(features)  # (N, n_data_points)
        pv = th.cumsum(intervals, dim=-1).clamp(0, 1).unsqueeze(1)  # (N, 1, n_data_points)

        # Compute Bernstein basis polynomials B_{i,n}(t)
        # B_{i,n}(t) = C(n,i) * t^i * (1-t)^(n-i)
        # where C(n,i) = n! / (i! * (n-i)!)

        device = features.device
        i = th.arange(0.0, self.n_control_points, device=device).view(1, -1, 1)
        n_minus_i = th.flip(i, dims=(1,))

        # Log-space computation for numerical stability
        # log(B_{i,n}(t)) = i*log(t) + (n-i)*log(1-t) + log(C(n,i))
        log_bernstein = (
            i * th.log(pv + self.eps)
            + n_minus_i * th.log(1 - pv + self.eps)
            + th.lgamma(th.tensor(self.n_control_points, device=device) + self.eps).view(1, -1, 1)
            - th.lgamma(i + 1 + self.eps)
            - th.lgamma(n_minus_i + 1 + self.eps)
        )
        bernstein = th.exp(log_bernstein)  # (N, n_control_points, n_data_points)

        # Rational Bezier curve: sum(w_i * P_i * B_i) / sum(w_i * B_i)
        numerator = (control_points * weights) @ bernstein  # (N, 2, n_data_points)
        denominator = (weights @ bernstein) + self.eps  # (N, 1, n_data_points)
        data_points = numerator / denominator

        return data_points, pv, intervals


class SNBezierDecoder(nn.Module):
    """Spectrally normalized Bezier decoder for airfoil generation.

    Generates airfoil coordinates and angle of attack from a latent code using
    rational Bezier curves. All components are spectrally normalized to ensure
    the overall network has Lipschitz constant ≤ lipschitz_scale.

    This decoder generates two outputs:
        - Bezier curve coordinates: (2, n_data_points) - x,y coordinates
        - Scalar output: single value (e.g., angle of attack)

    Architecture:
        1. Feature generator (SNMLP): latent → features for Bezier layer
        2. CPW generator (SNMLP + deconv): latent → control points & weights
        3. Bezier layer: features + CP + W → curve coordinates
        4. Scalar head (SN Linear): latent → scalar output
        5. Scale layer: scale entire output by lipschitz_scale

    Args:
        latent_dim: Dimensionality of input latent code
        n_control_points: Number of Bezier control points
        n_data_points: Number of output points along curve (e.g., 192)
        lipschitz_scale: Global Lipschitz constant multiplier (default: 1.0)
        m_features: Intermediate feature size for Bezier layer (default: 256)
        feature_gen_hidden: Hidden layers for feature generator (default: [1024, 512])
        cpw_mlp_hidden: Hidden layers for CPW generator MLP (default: [1024])
        cpw_deconv_channels: Channel progression for CPW deconv (default: [768, 384, 192, 96])

    Shape:
        - Input: (N, latent_dim)
        - Output coords: (N, 2, n_data_points)
        - Output scalar: (N, 1)

    Example:
        >>> decoder = SNBezierDecoder(
        ...     latent_dim=100,
        ...     n_control_points=64,
        ...     n_data_points=192,
        ...     lipschitz_scale=1.0
        ... )
        >>> coords, scalar = decoder(z)  # z shape: (B, 100)

    Note:
        The Lipschitz constant of the overall network is bounded by lipschitz_scale
        assuming:
        - All linear/conv layers are spectrally normalized (Lipschitz ≤ 1)
        - Activations (LeakyReLU, Tanh, Sigmoid) have Lipschitz ≤ 1
        - The Bezier layer has bounded Lipschitz constant
        - No BatchNorm or other statistics-dependent normalization is used

        Normalization/denormalization of outputs should be handled externally
        using a Normalizer class or similar.
    """

    def __init__(
        self,
        latent_dim: int,
        n_control_points: int,
        n_data_points: int,
        lipschitz_scale: float = 1.0,
        m_features: int = 256,
        feature_gen_hidden: list[int] | None = None,
        cpw_mlp_hidden: list[int] | None = None,
        cpw_deconv_channels: list[int] | None = None,
    ):
        super().__init__()

        # Default hyperparameters
        if feature_gen_hidden is None:
            feature_gen_hidden = [1024, 512]
        if cpw_mlp_hidden is None:
            cpw_mlp_hidden = [1024]
        if cpw_deconv_channels is None:
            cpw_deconv_channels = [768, 384, 192, 96]

        self.latent_dim = latent_dim
        self.n_control_points = n_control_points
        self.n_data_points = n_data_points

        # 1. Feature generator for Bezier layer (SNMLP)
        self.feature_generator = SNMLP(
            in_features=latent_dim,
            out_features=m_features,
            hidden_widths=feature_gen_hidden,
        )

        # 2. Control Point and Weight (CPW) generator
        # 2a. Calculate required deconv input dimensions
        n_deconv_layers = len(cpw_deconv_channels) - 1
        self.cpw_in_width = n_control_points // (2 ** n_deconv_layers)
        self.cpw_in_channels = cpw_deconv_channels[0]

        # 2b. MLP to generate flattened deconv input
        self.cpw_dense = SNMLP(
            in_features=latent_dim,
            out_features=self.cpw_in_channels * self.cpw_in_width,
            hidden_widths=cpw_mlp_hidden,
        )

        # 2c. Deconvolutional layers (spectrally normalized)
        deconv_layers: list[nn.Module] = []
        for i in range(len(cpw_deconv_channels) - 1):
            # ConvTranspose1d: kernel=4, stride=2, padding=1 (doubles spatial size)
            deconv = spectral_norm(
                nn.ConvTranspose1d(
                    cpw_deconv_channels[i],
                    cpw_deconv_channels[i + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            deconv_layers.append(deconv)
            deconv_layers.append(nn.LeakyReLU(0.2))  # Lipschitz ≤ 1

        self.deconv = nn.Sequential(*deconv_layers)

        # 2d. Output heads for control points and weights
        self.cp_gen = nn.Sequential(
            spectral_norm(nn.Conv1d(cpw_deconv_channels[-1], 2, kernel_size=1)),
            nn.Sigmoid(),  # Lipschitz ≤ 1, output in [0, 1]
        )
        self.w_gen = nn.Sequential(
            spectral_norm(nn.Conv1d(cpw_deconv_channels[-1], 1, kernel_size=1)),
            nn.Sigmoid(),  # Lipschitz ≤ 1, output in [0, 1]
        )

        # 3. Bezier layer (with SN on interval generation)
        self.bezier_layer = BezierLayer(
            m_features=m_features,
            n_control_points=n_control_points,
            n_data_points=n_data_points,
            use_spectral_norm=True,
        )

        # 4. Scalar head (spectrally normalized)
        self.scalar_head = spectral_norm(nn.Linear(latent_dim, 1))

        # 5. Final scaling layer
        self.scale = Scale(lipschitz_scale)

    def forward(self, z: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Generate curve coordinates and scalar output from latent code.

        Args:
            z: Latent code (N, latent_dim)

        Returns:
            Tuple of:
                - coords: Curve coordinates (N, 2, n_data_points)
                - scalar: Scalar output (N, 1)
        """
        # Generate features for Bezier layer
        features = self.feature_generator(z)

        # Generate control points and weights
        x = self.cpw_dense(z).view(-1, self.cpw_in_channels, self.cpw_in_width)
        x = self.deconv(x)
        cp = self.cp_gen(x)  # (N, 2, n_control_points)
        w = self.w_gen(x)    # (N, 1, n_control_points)

        # Generate Bezier curve
        coords, _, _ = self.bezier_layer(features, cp, w)

        # Generate scalar output (passed through sigmoid for [0, 1] range)
        scalar = th.sigmoid(self.scalar_head(z))  # (N, 1) in [0, 1]

        # Apply global Lipschitz scaling
        coords = self.scale(coords)
        scalar = self.scale(scalar)

        return coords, scalar


class FactorizedConv1DEncoder(nn.Module):
    """Factorized 1D convolutional encoder for sequences with auxiliary scalar input.

    Encodes a primary sequence (e.g., airfoil coordinates) and an auxiliary scalar
    (e.g., angle of attack) into separate latent dimensions, then concatenates them.

    Architecture:
        - Primary path: Conv1D → MLP → z_primary (latent_dim - n_aux dimensions)
        - Auxiliary path: Linear → z_aux (n_aux dimensions)
        - Output: z = [z_primary | z_aux] (latent_dim dimensions)

    This factorization allows:
    - Excluding auxiliary dimensions from volume loss
    - Protecting auxiliary dimensions from pruning
    - Separate decoder paths for primary and auxiliary outputs

    Args:
        in_channels: Number of input channels for primary sequence
        in_features: Length of input sequence
        latent_dim: Total dimensionality of latent code
        n_aux: Number of auxiliary latent dimensions (default: 1)
        conv_channels: List of channel sizes for conv layers
        mlp_hidden: List of hidden layer widths for final MLP
        use_spectral_norm: Whether to apply spectral normalization (default: True)

    Shape:
        - Input primary: (N, in_channels, in_features)
        - Input auxiliary: (N, n_aux)
        - Output: (N, latent_dim) where z = [z_primary | z_aux]

    Example:
        >>> encoder = FactorizedConv1DEncoder(
        ...     in_channels=2,
        ...     in_features=192,
        ...     latent_dim=100,
        ...     n_aux=1,
        ...     conv_channels=[64, 128, 256],
        ...     mlp_hidden=[512, 256],
        ...     use_spectral_norm=True
        ... )
        >>> z = encoder(coords, angle)  # coords: (B, 2, 192), angle: (B, 1)
    """

    def __init__(
        self,
        in_channels: int,
        in_features: int,
        latent_dim: int,
        n_aux: int = 1,
        conv_channels: list[int] | None = None,
        mlp_hidden: list[int] | None = None,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_aux = n_aux
        self.latent_dim_primary = latent_dim - n_aux

        if conv_channels is None:
            conv_channels = [64, 128, 256, 512, 1024, 2048]
        if mlp_hidden is None:
            mlp_hidden = [1024, 512]

        # Primary encoder: Use Conv1DEncoder or SNConv1DEncoder
        if use_spectral_norm:
            self.primary_encoder = SNConv1DEncoder(
                in_channels=in_channels,
                in_features=in_features,
                latent_dim=self.latent_dim_primary,
                conv_channels=conv_channels,
                mlp_hidden=mlp_hidden,
            )
        else:
            self.primary_encoder = Conv1DEncoder(
                in_channels=in_channels,
                in_features=in_features,
                latent_dim=self.latent_dim_primary,
                conv_channels=conv_channels,
                mlp_hidden=mlp_hidden,
                use_spectral_norm=False,
            )

        # Auxiliary encoder: Simple linear layer
        aux_linear = nn.Linear(n_aux, n_aux)
        if use_spectral_norm:
            aux_linear = spectral_norm(aux_linear)
        self.aux_encoder = aux_linear

    def forward(self, primary: th.Tensor, auxiliary: th.Tensor) -> th.Tensor:
        """Encode inputs with factorized latent space.

        Args:
            primary: Primary input sequence (N, in_channels, in_features)
            auxiliary: Auxiliary scalar input (N, n_aux)

        Returns:
            z: Factorized latent code (N, latent_dim) = [z_primary | z_aux]
        """
        # Encode primary → z_primary
        z_primary = self.primary_encoder(primary)  # (N, latent_dim - n_aux)

        # Encode auxiliary → z_aux
        z_aux = self.aux_encoder(auxiliary)  # (N, n_aux)

        # Concatenate: [z_primary | z_aux]
        return th.cat([z_primary, z_aux], dim=1)  # (N, latent_dim)


class FactorizedBezierDecoder(nn.Module):
    """Factorized Bezier decoder for generating curves with auxiliary scalar output.

    Decodes a factorized latent code into:
    - Primary output: Bezier curve coordinates (from z_primary)
    - Auxiliary output: Scalar value (from z_aux)

    Architecture:
        - Primary path: z_primary → SNBezierDecoder → curve coordinates
        - Auxiliary path: z_aux → Linear → scalar output
        - No cross-dependencies between paths

    This factorization ensures explicit separation between primary geometry
    generation and auxiliary parameter generation.

    Args:
        latent_dim: Total dimensionality of input latent code
        n_aux: Number of auxiliary latent dimensions (default: 1)
        n_control_points: Number of Bezier control points
        n_data_points: Number of output points along curve
        lipschitz_scale: Global Lipschitz constant multiplier (default: 1.0)
        m_features: Intermediate feature size for Bezier layer (default: 256)
        feature_gen_hidden: Hidden layers for feature generator
        cpw_mlp_hidden: Hidden layers for CPW generator MLP
        cpw_deconv_channels: Channel progression for CPW deconv

    Shape:
        - Input: (N, latent_dim) where z = [z_primary | z_aux]
        - Output coords: (N, 2, n_data_points)
        - Output auxiliary: (N, n_aux)

    Example:
        >>> decoder = FactorizedBezierDecoder(
        ...     latent_dim=100,
        ...     n_aux=1,
        ...     n_control_points=64,
        ...     n_data_points=192,
        ...     lipschitz_scale=1.0
        ... )
        >>> coords, angle = decoder(z)  # z: (B, 100)
    """

    def __init__(
        self,
        latent_dim: int,
        n_aux: int = 1,
        n_control_points: int = 64,
        n_data_points: int = 192,
        lipschitz_scale: float = 1.0,
        m_features: int = 256,
        feature_gen_hidden: list[int] | None = None,
        cpw_mlp_hidden: list[int] | None = None,
        cpw_deconv_channels: list[int] | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_aux = n_aux
        self.latent_dim_primary = latent_dim - n_aux

        # Primary decoder: Use SNBezierDecoder for curve generation
        self.primary_decoder = SNBezierDecoder(
            latent_dim=self.latent_dim_primary,
            n_control_points=n_control_points,
            n_data_points=n_data_points,
            lipschitz_scale=lipschitz_scale,
            m_features=m_features,
            feature_gen_hidden=feature_gen_hidden,
            cpw_mlp_hidden=cpw_mlp_hidden,
            cpw_deconv_channels=cpw_deconv_channels,
        )

        # Auxiliary decoder: Simple linear layer with scaling
        self.aux_head = spectral_norm(nn.Linear(n_aux, n_aux))
        self.aux_scale = Scale(lipschitz_scale)

    def forward(self, z: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Generate curve coordinates and auxiliary output from factorized latent code.

        Args:
            z: Latent code (N, latent_dim) factorized as [z_primary | z_aux]

        Returns:
            Tuple of:
                - coords: Curve coordinates (N, 2, n_data_points)
                - auxiliary: Auxiliary output (N, n_aux) in [0, 1] range
        """
        # Split latent into primary and auxiliary
        z_primary = z[:, : -self.n_aux]  # (N, latent_dim - n_aux)
        z_aux = z[:, -self.n_aux :]  # (N, n_aux)

        # Generate coords from z_primary using SNBezierDecoder
        # Note: SNBezierDecoder returns (coords, scalar), we only want coords
        coords, _ = self.primary_decoder(z_primary)

        # Generate auxiliary from z_aux using simple linear layer
        auxiliary = th.sigmoid(self.aux_head(z_aux))  # (N, n_aux) in [0, 1]
        auxiliary = self.aux_scale(auxiliary)

        return coords, auxiliary
