"""Constraint optimization methods for LVAE training.

This module provides a modular framework for handling constrained optimization
in LVAE models. Different constraint handling methods can be easily swapped via
command-line arguments.

Problem formulation:
    minimize: volume_loss
    subject to: reconstruction_loss ≤ reconstruction_threshold
                performance_loss ≤ performance_threshold

Available methods:
    - WeightedSumHandler: Simple weighted combination (no hard constraints)
    - AugmentedLagrangianHandler: Augmented Lagrangian with penalty terms
    - LogBarrierHandler: Interior point method with logarithmic barriers
    - PrimalDualHandler: Primal-dual gradient method (simplified Lagrangian)
    - AdaptiveWeightHandler: Automatic weight balancing via gradient normalization
    - SoftplusALHandler: Smoothed augmented Lagrangian using softplus
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ConstraintLosses:
    """Container for loss components."""

    volume: torch.Tensor
    reconstruction: torch.Tensor
    performance: torch.Tensor


@dataclass
class ConstraintThresholds:
    """Container for constraint thresholds."""

    reconstruction: float
    performance: float


class ConstraintHandler(ABC):
    """Abstract base class for constraint optimization methods.

    All constraint handlers must implement:
        - compute_loss(): Combines loss components according to method
        - step(): Updates method-specific state (e.g., dual variables, barrier parameter)
        - get_metrics(): Returns logging metrics for wandb
    """

    def __init__(self, device: torch.device = torch.device("cpu"), volume_warmup_epochs: int = 0):
        """Initialize constraint handler.

        Args:
            device: Device to store tensors on
            volume_warmup_epochs: Number of epochs to ignore volume loss (default: 0)
        """
        self.device = device
        self.volume_warmup_epochs = volume_warmup_epochs
        self._epoch = 0

    @property
    def _volume_warmup_active(self) -> bool:
        """Check if volume warmup is active (volume loss should be ignored)."""
        return self._epoch < self.volume_warmup_epochs

    @abstractmethod
    def compute_loss(
        self,
        losses: ConstraintLosses,
        thresholds: ConstraintThresholds,
    ) -> torch.Tensor:
        """Compute total loss from components.

        Args:
            losses: Loss components (volume, reconstruction, performance)
            thresholds: Constraint thresholds

        Returns:
            Total loss for optimization
        """
        pass

    @abstractmethod
    def step(self, losses: ConstraintLosses, thresholds: ConstraintThresholds) -> None:
        """Update method-specific state after optimization step.

        Called once per batch/epoch to update dual variables, barrier parameters,
        adaptive weights, etc.

        Args:
            losses: Current loss values (detached)
            thresholds: Constraint thresholds
        """
        pass

    def epoch_hook(self, epoch: int) -> None:
        """Called at the start of each epoch.

        Args:
            epoch: Current epoch number
        """
        self._epoch = epoch

    @abstractmethod
    def get_metrics(self) -> dict[str, float]:
        """Get loggable metrics for current state.

        Returns:
            Dictionary of metric names to values for wandb logging
        """
        pass


class WeightedSumHandler(ConstraintHandler):
    """Simple weighted sum of all objectives (no hard constraints).

    Loss = w_v * volume + w_r * reconstruction + w_p * performance

    This is the method used in lvae_1d. Weights control relative importance
    but don't enforce hard constraints.

    Args:
        w_volume: Weight for volume loss (default: 1.0)
        w_reconstruction: Weight for reconstruction loss (default: 1.0)
        w_performance: Weight for performance loss (default: 1.0)
    """

    def __init__(
        self,
        w_volume: float = 1.0,
        w_reconstruction: float = 1.0,
        w_performance: float = 1.0,
        device: torch.device = torch.device("cpu"),
        volume_warmup_epochs: int = 0,
    ):
        super().__init__(device, volume_warmup_epochs)
        self.w_v = w_volume
        self.w_r = w_reconstruction
        self.w_p = w_performance

    def compute_loss(
        self,
        losses: ConstraintLosses,
        thresholds: ConstraintThresholds,
    ) -> torch.Tensor:
        # During warmup, only minimize reconstruction + performance
        if self._volume_warmup_active:
            return self.w_r * losses.reconstruction + self.w_p * losses.performance
        # After warmup, include volume loss
        return self.w_v * losses.volume + self.w_r * losses.reconstruction + self.w_p * losses.performance

    def step(self, losses: ConstraintLosses, thresholds: ConstraintThresholds) -> None:
        pass  # No state to update

    def get_metrics(self) -> dict[str, float]:
        return {
            "constraint/w_volume": self.w_v,
            "constraint/w_reconstruction": self.w_r,
            "constraint/w_performance": self.w_p,
        }


class AugmentedLagrangianHandler(ConstraintHandler):
    """Augmented Lagrangian method with quadratic penalty terms.

    Loss = volume
           + λ_r * max(0, rec - ε_r) + (μ_r/2) * max(0, rec - ε_r)²
           + λ_p * max(0, perf - ε_p) + (μ_p/2) * max(0, perf - ε_p)²

    Lagrange multipliers (λ) are updated via gradient ascent:
        λ ← max(0, λ + α * violation)

    Args:
        mu_r_init: Initial quadratic penalty coefficient for reconstruction (default: 1.0)
        mu_p_init: Initial quadratic penalty coefficient for performance (default: 1.0)
        mu_r_final: Final quadratic penalty coefficient for reconstruction (default: 10.0)
        mu_p_final: Final quadratic penalty coefficient for performance (default: 10.0)
        alpha_r: Learning rate for reconstruction multiplier (default: 0.1)
        alpha_p: Learning rate for performance multiplier (default: 0.1)
        warmup_epochs: Epochs to linearly ramp up penalty coefficients (default: 100)
    """

    def __init__(
        self,
        mu_r_init: float = 1.0,
        mu_p_init: float = 1.0,
        mu_r_final: float = 10.0,
        mu_p_final: float = 10.0,
        alpha_r: float = 0.1,
        alpha_p: float = 0.1,
        warmup_epochs: int = 100,
        device: torch.device = torch.device("cpu"),
        volume_warmup_epochs: int = 0,
    ):
        super().__init__(device, volume_warmup_epochs)
        self.mu_r_init = mu_r_init
        self.mu_p_init = mu_p_init
        self.mu_r_final = mu_r_final
        self.mu_p_final = mu_p_final
        self.alpha_r = alpha_r
        self.alpha_p = alpha_p
        self.warmup_epochs = warmup_epochs

        # State variables
        self.lambda_r = torch.tensor(0.0, device=device)
        self.lambda_p = torch.tensor(0.0, device=device)

    def _get_penalty_coefficients(self) -> tuple[float, float]:
        """Get current penalty coefficients based on warmup schedule."""
        if self._epoch < self.warmup_epochs:
            warmup_factor = self._epoch / self.warmup_epochs
            mu_r = self.mu_r_init + (self.mu_r_final - self.mu_r_init) * warmup_factor
            mu_p = self.mu_p_init + (self.mu_p_final - self.mu_p_init) * warmup_factor
        else:
            mu_r = self.mu_r_final
            mu_p = self.mu_p_final
        return mu_r, mu_p

    def compute_loss(
        self,
        losses: ConstraintLosses,
        thresholds: ConstraintThresholds,
    ) -> torch.Tensor:
        # During volume warmup, only minimize reconstruction + performance
        if self._volume_warmup_active:
            return losses.reconstruction + losses.performance

        mu_r, mu_p = self._get_penalty_coefficients()

        # Compute violations
        rec_violation = torch.clamp(losses.reconstruction - thresholds.reconstruction, min=0.0)
        perf_violation = torch.clamp(losses.performance - thresholds.performance, min=0.0)

        # Augmented Lagrangian formulation
        total_loss = (
            losses.volume
            + self.lambda_r * rec_violation
            + 0.5 * mu_r * rec_violation**2
            + self.lambda_p * perf_violation
            + 0.5 * mu_p * perf_violation**2
        )

        return total_loss

    def step(self, losses: ConstraintLosses, thresholds: ConstraintThresholds) -> None:
        """Update Lagrange multipliers via gradient ascent."""
        rec_violation = torch.clamp(losses.reconstruction - thresholds.reconstruction, min=0.0)
        perf_violation = torch.clamp(losses.performance - thresholds.performance, min=0.0)

        # Gradient ascent on dual variables (project to non-negative orthant)
        self.lambda_r = torch.clamp(self.lambda_r + self.alpha_r * rec_violation, min=0.0)
        self.lambda_p = torch.clamp(self.lambda_p + self.alpha_p * perf_violation, min=0.0)

    def get_metrics(self) -> dict[str, float]:
        mu_r, mu_p = self._get_penalty_coefficients()
        return {
            "constraint/lambda_r": self.lambda_r.item(),
            "constraint/lambda_p": self.lambda_p.item(),
            "constraint/mu_r": mu_r,
            "constraint/mu_p": mu_p,
        }


class LogBarrierHandler(ConstraintHandler):
    """Interior point method using logarithmic barrier functions.

    Loss = volume - (1/t) * [log(ε_r - rec) + log(ε_p - perf)]

    The barrier prevents constraint violations by construction (loss → ∞ as we
    approach constraint boundary). Parameter t increases over training to tighten
    constraint enforcement.

    Advantages:
        - Smooth gradients everywhere in feasible region
        - Single hyperparameter (t) to tune
        - No multiplier updates needed
        - Proven superior for constrained CNNs

    Disadvantages:
        - Requires starting with feasible solution (losses < thresholds)
        - Numerical instability near boundaries

    Args:
        t_init: Initial barrier parameter (default: 1.0, larger = stricter)
        t_growth: Multiplicative growth rate per epoch (default: 1.05)
        t_max: Maximum barrier parameter (default: 1000.0)
        epsilon: Safety margin from boundary (default: 1e-6)
        fallback_penalty: Penalty multiplier when infeasible (default: 1e6)
    """

    def __init__(
        self,
        t_init: float = 1.0,
        t_growth: float = 1.05,
        t_max: float = 1000.0,
        epsilon: float = 1e-6,
        fallback_penalty: float = 1e6,
        device: torch.device = torch.device("cpu"),
        volume_warmup_epochs: int = 0,
    ):
        super().__init__(device, volume_warmup_epochs)
        self.t = torch.tensor(t_init, device=device)
        self.t_growth = t_growth
        self.t_max = t_max
        self.epsilon = epsilon
        self.fallback_penalty = fallback_penalty

    def compute_loss(
        self,
        losses: ConstraintLosses,
        thresholds: ConstraintThresholds,
    ) -> torch.Tensor:
        # During volume warmup, only minimize reconstruction + performance
        if self._volume_warmup_active:
            return losses.reconstruction + losses.performance

        # Compute slack (how far we are from constraint boundary)
        rec_slack = thresholds.reconstruction - losses.reconstruction - self.epsilon
        perf_slack = thresholds.performance - losses.performance - self.epsilon

        # Check feasibility
        if rec_slack <= 0 or perf_slack <= 0:
            # Infeasible: fall back to large penalty
            rec_violation = torch.clamp(-rec_slack, min=0.0)
            perf_violation = torch.clamp(-perf_slack, min=0.0)
            return losses.volume + self.fallback_penalty * (rec_violation + perf_violation)

        # Feasible: use log barrier
        barrier_term = -(1.0 / self.t) * (torch.log(rec_slack) + torch.log(perf_slack))
        return losses.volume + barrier_term

    def step(self, losses: ConstraintLosses, thresholds: ConstraintThresholds) -> None:
        """Increase barrier parameter to tighten constraints."""
        self.t = torch.clamp(self.t * self.t_growth, max=self.t_max)

    def get_metrics(self) -> dict[str, float]:
        return {
            "constraint/t": self.t.item(),
            "constraint/barrier_strength": 1.0 / self.t.item(),
        }


class PrimalDualHandler(ConstraintHandler):
    """Primal-dual gradient method (simplified Lagrangian).

    Loss = volume + λ_r * (rec - ε_r) + λ_p * (perf - ε_p)

    Unlike augmented Lagrangian, this uses only linear penalty terms (no quadratic).
    Dual variables (λ) are updated via gradient ascent:
        λ ← max(0, λ + lr_dual * violation)

    Advantages:
        - No penalty parameters (μ) needed
        - Better convergence theory for saddle point problems
        - Natural constraint balancing
        - Simple implementation

    Args:
        lr_dual: Learning rate for dual variable updates (default: 0.01)
        clip_lambda: Maximum value for dual variables (default: 100.0)
    """

    def __init__(
        self,
        lr_dual: float = 0.01,
        clip_lambda: float = 100.0,
        device: torch.device = torch.device("cpu"),
        volume_warmup_epochs: int = 0,
    ):
        super().__init__(device, volume_warmup_epochs)
        self.lr_dual = lr_dual
        self.clip_lambda = clip_lambda

        # Dual variables (Lagrange multipliers)
        self.lambda_r = torch.tensor(0.0, device=device)
        self.lambda_p = torch.tensor(0.0, device=device)

    def compute_loss(
        self,
        losses: ConstraintLosses,
        thresholds: ConstraintThresholds,
    ) -> torch.Tensor:
        # During volume warmup, only minimize reconstruction + performance
        if self._volume_warmup_active:
            return losses.reconstruction + losses.performance

        # Simple Lagrangian (linear penalty only)
        rec_violation = losses.reconstruction - thresholds.reconstruction
        perf_violation = losses.performance - thresholds.performance

        total_loss = losses.volume + self.lambda_r * rec_violation + self.lambda_p * perf_violation

        return total_loss

    def step(self, losses: ConstraintLosses, thresholds: ConstraintThresholds) -> None:
        """Update dual variables via gradient ascent."""
        rec_violation = losses.reconstruction - thresholds.reconstruction
        perf_violation = losses.performance - thresholds.performance

        # Gradient ascent on dual (project to [0, clip_lambda])
        self.lambda_r = torch.clamp(self.lambda_r + self.lr_dual * rec_violation, min=0.0, max=self.clip_lambda)
        self.lambda_p = torch.clamp(self.lambda_p + self.lr_dual * perf_violation, min=0.0, max=self.clip_lambda)

    def get_metrics(self) -> dict[str, float]:
        return {
            "constraint/lambda_r": self.lambda_r.item(),
            "constraint/lambda_p": self.lambda_p.item(),
        }


class AdaptiveWeightHandler(ConstraintHandler):
    """Automatic weight balancing using gradient normalization (GradNorm-inspired).

    Loss = w_v * volume + w_r * reconstruction + w_p * performance

    Weights adapt automatically to balance gradient magnitudes across objectives.
    Target: all loss terms should have similar influence on parameter updates.

    Weight update rule:
        w_i ← w_i * exp(α * (||∇_θ L_i|| / mean(||∇_θ L_j||) - 1))

    Advantages:
        - Automatic balancing, no manual tuning
        - Responds to training dynamics
        - Prevents one objective from dominating

    Disadvantages:
        - No hard constraint enforcement
        - Additional computational cost (gradient norm computation)
        - May need tuning of adaptation rate

    Args:
        w_volume_init: Initial weight for volume (default: 1.0)
        w_reconstruction_init: Initial weight for reconstruction (default: 1.0)
        w_performance_init: Initial weight for performance (default: 1.0)
        adaptation_lr: Learning rate for weight updates (default: 0.01)
        update_frequency: Update weights every N steps (default: 10)
    """

    def __init__(
        self,
        w_volume_init: float = 1.0,
        w_reconstruction_init: float = 1.0,
        w_performance_init: float = 1.0,
        adaptation_lr: float = 0.01,
        update_frequency: int = 10,
        device: torch.device = torch.device("cpu"),
        volume_warmup_epochs: int = 0,
    ):
        super().__init__(device, volume_warmup_epochs)
        # Store weights in log-space for numerical stability
        self.log_w_v = torch.tensor(torch.log(torch.tensor(w_volume_init)), device=device)
        self.log_w_r = torch.tensor(torch.log(torch.tensor(w_reconstruction_init)), device=device)
        self.log_w_p = torch.tensor(torch.log(torch.tensor(w_performance_init)), device=device)

        self.adaptation_lr = adaptation_lr
        self.update_frequency = update_frequency
        self._step_count = 0

        # Cache for gradient norms
        self._grad_norms: dict[str, float] | None = None

    def compute_loss(
        self,
        losses: ConstraintLosses,
        thresholds: ConstraintThresholds,
    ) -> torch.Tensor:
        # During volume warmup, only minimize reconstruction + performance
        if self._volume_warmup_active:
            w_r = torch.exp(self.log_w_r)
            w_p = torch.exp(self.log_w_p)
            return w_r * losses.reconstruction + w_p * losses.performance

        # Convert from log-space
        w_v = torch.exp(self.log_w_v)
        w_r = torch.exp(self.log_w_r)
        w_p = torch.exp(self.log_w_p)

        return w_v * losses.volume + w_r * losses.reconstruction + w_p * losses.performance

    def step(self, losses: ConstraintLosses, thresholds: ConstraintThresholds) -> None:
        """Update weights based on cached gradient norms."""
        self._step_count += 1

        # Only update weights periodically
        if self._step_count % self.update_frequency != 0:
            return

        # If gradient norms were computed, adapt weights
        if self._grad_norms is not None:
            norm_v = self._grad_norms["volume"]
            norm_r = self._grad_norms["reconstruction"]
            norm_p = self._grad_norms["performance"]

            # Target: all gradients should have similar magnitude
            mean_norm = (norm_v + norm_r + norm_p) / 3.0

            # Update weights to equalize gradient magnitudes
            # If norm_i < mean, increase w_i; if norm_i > mean, decrease w_i
            if mean_norm > 1e-8:  # Avoid division by zero
                self.log_w_v += self.adaptation_lr * (1.0 - norm_v / mean_norm)
                self.log_w_r += self.adaptation_lr * (1.0 - norm_r / mean_norm)
                self.log_w_p += self.adaptation_lr * (1.0 - norm_p / mean_norm)

            self._grad_norms = None

    def set_gradient_norms(self, norm_volume: float, norm_reconstruction: float, norm_performance: float) -> None:
        """Cache gradient norms for weight adaptation.

        This should be called after backward() but before step().

        Args:
            norm_volume: L2 norm of volume loss gradient
            norm_reconstruction: L2 norm of reconstruction loss gradient
            norm_performance: L2 norm of performance loss gradient
        """
        self._grad_norms = {
            "volume": norm_volume,
            "reconstruction": norm_reconstruction,
            "performance": norm_performance,
        }

    def get_metrics(self) -> dict[str, float]:
        metrics = {
            "constraint/w_volume": torch.exp(self.log_w_v).item(),
            "constraint/w_reconstruction": torch.exp(self.log_w_r).item(),
            "constraint/w_performance": torch.exp(self.log_w_p).item(),
        }

        # Include gradient norms if available
        if self._grad_norms is not None:
            metrics.update({
                "constraint/grad_norm_volume": self._grad_norms["volume"],
                "constraint/grad_norm_reconstruction": self._grad_norms["reconstruction"],
                "constraint/grad_norm_performance": self._grad_norms["performance"],
            })

        return metrics


class SoftplusALHandler(ConstraintHandler):
    """Augmented Lagrangian with smooth softplus activation instead of ReLU.

    Loss = volume
           + λ_r * softplus(rec - ε_r, β) + (μ_r/2) * softplus(rec - ε_r, β)²
           + λ_p * softplus(perf - ε_p, β) + (μ_p/2) * softplus(perf - ε_p, β)²

    Replaces max(0, violation) with softplus for better gradient flow:
        softplus(x, β) = (1/β) * log(1 + exp(β * x))

    As β → ∞, softplus → ReLU, but finite β provides smooth gradients.

    Advantages:
        - Smooth gradients everywhere (no zero gradient region)
        - Otherwise identical to augmented Lagrangian
        - Easy drop-in replacement

    Args:
        beta: Smoothness parameter for softplus (default: 10.0, larger = sharper)
        mu_r_init, mu_p_init, mu_r_final, mu_p_final: Penalty coefficients
        alpha_r, alpha_p: Multiplier learning rates
        warmup_epochs: Penalty coefficient warmup duration
    """

    def __init__(
        self,
        beta: float = 10.0,
        mu_r_init: float = 1.0,
        mu_p_init: float = 1.0,
        mu_r_final: float = 10.0,
        mu_p_final: float = 10.0,
        alpha_r: float = 0.1,
        alpha_p: float = 0.1,
        warmup_epochs: int = 100,
        device: torch.device = torch.device("cpu"),
        volume_warmup_epochs: int = 0,
    ):
        super().__init__(device, volume_warmup_epochs)
        self.beta = beta
        self.mu_r_init = mu_r_init
        self.mu_p_init = mu_p_init
        self.mu_r_final = mu_r_final
        self.mu_p_final = mu_p_final
        self.alpha_r = alpha_r
        self.alpha_p = alpha_p
        self.warmup_epochs = warmup_epochs

        # State variables
        self.lambda_r = torch.tensor(0.0, device=device)
        self.lambda_p = torch.tensor(0.0, device=device)

    def _get_penalty_coefficients(self) -> tuple[float, float]:
        """Get current penalty coefficients based on warmup schedule."""
        if self._epoch < self.warmup_epochs:
            warmup_factor = self._epoch / self.warmup_epochs
            mu_r = self.mu_r_init + (self.mu_r_final - self.mu_r_init) * warmup_factor
            mu_p = self.mu_p_init + (self.mu_p_final - self.mu_p_init) * warmup_factor
        else:
            mu_r = self.mu_r_final
            mu_p = self.mu_p_final
        return mu_r, mu_p

    def compute_loss(
        self,
        losses: ConstraintLosses,
        thresholds: ConstraintThresholds,
    ) -> torch.Tensor:
        # During volume warmup, only minimize reconstruction + performance
        if self._volume_warmup_active:
            return losses.reconstruction + losses.performance

        mu_r, mu_p = self._get_penalty_coefficients()

        # Compute smooth violations using softplus
        rec_violation_raw = losses.reconstruction - thresholds.reconstruction
        perf_violation_raw = losses.performance - thresholds.performance

        rec_violation = F.softplus(rec_violation_raw, beta=self.beta)
        perf_violation = F.softplus(perf_violation_raw, beta=self.beta)

        # Augmented Lagrangian with smooth violations
        total_loss = (
            losses.volume
            + self.lambda_r * rec_violation
            + 0.5 * mu_r * rec_violation**2
            + self.lambda_p * perf_violation
            + 0.5 * mu_p * perf_violation**2
        )

        return total_loss

    def step(self, losses: ConstraintLosses, thresholds: ConstraintThresholds) -> None:
        """Update Lagrange multipliers via gradient ascent."""
        rec_violation_raw = losses.reconstruction - thresholds.reconstruction
        perf_violation_raw = losses.performance - thresholds.performance

        # Use softplus for smooth multiplier updates
        rec_violation = F.softplus(rec_violation_raw, beta=self.beta)
        perf_violation = F.softplus(perf_violation_raw, beta=self.beta)

        # Gradient ascent on dual variables (project to non-negative orthant)
        self.lambda_r = torch.clamp(self.lambda_r + self.alpha_r * rec_violation, min=0.0)
        self.lambda_p = torch.clamp(self.lambda_p + self.alpha_p * perf_violation, min=0.0)

    def get_metrics(self) -> dict[str, float]:
        mu_r, mu_p = self._get_penalty_coefficients()
        return {
            "constraint/lambda_r": self.lambda_r.item(),
            "constraint/lambda_p": self.lambda_p.item(),
            "constraint/mu_r": mu_r,
            "constraint/mu_p": mu_p,
            "constraint/beta": self.beta,
        }


# Factory function for easy instantiation
def create_constraint_handler(
    method: str,
    device: torch.device = torch.device("cpu"),
    **kwargs,
) -> ConstraintHandler:
    """Factory function to create constraint handler by name.

    Args:
        method: Handler name (weighted_sum, augmented_lagrangian, log_barrier,
                primal_dual, adaptive, softplus_al)
        device: Device to store tensors on
        **kwargs: Method-specific arguments

    Returns:
        Instantiated constraint handler

    Raises:
        ValueError: If method name is unknown
    """
    handlers = {
        "weighted_sum": WeightedSumHandler,
        "augmented_lagrangian": AugmentedLagrangianHandler,
        "log_barrier": LogBarrierHandler,
        "primal_dual": PrimalDualHandler,
        "adaptive": AdaptiveWeightHandler,
        "softplus_al": SoftplusALHandler,
    }

    if method not in handlers:
        raise ValueError(
            f"Unknown constraint method: {method}. " f"Available methods: {list(handlers.keys())}"
        )

    return handlers[method](device=device, **kwargs)
