"""The `Generator` contract: what a model implements to join the EngiOpt leaderboard.

A generator is *not* a training loop. Training scripts stay single-file and
self-contained (the CleanRL philosophy this repository is built on). A generator
is the thin, uniform surface the evaluator needs in order to treat every model
the same way:

1. `from_pretrained` -- rebuild a trained model from a checkpoint package.
2. `sample` -- turn conditions into designs.

Everything else (which metrics run, how conditions are drawn, how results are
stored) belongs to the evaluator, not the model.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from functools import cached_property
import hashlib
import json
import time
from typing import Any, ClassVar, TYPE_CHECKING

from gymnasium import spaces
import numpy as np
import torch as th

from engiopt.checkpoint_store import resolve_named_checkpoint
from engiopt.transforms import condition_keys as scalar_condition_keys
from engiopt.transforms import image_condition_keys as scalar_image_condition_keys

if TYPE_CHECKING:
    from datasets import Dataset
    from engibench.core import Problem
    import numpy.typing as npt

    from engiopt.checkpoint_store import ModelSource
    from engiopt.checkpoint_store import ResolvedCheckpoint

DEFAULT_HF_ENTITY = "IDEALLab"
DEFAULT_HF_REPO_PREFIX = "engiopt"


@dataclass(frozen=True)
class ConditionBatch:
    """The conditions a generator is asked to satisfy, in every form models need.

    A design requirement comes in two shapes, and they cannot share a tensor.
    Scalars (a volume budget, a filter radius) stack into `tensor`. Fields
    (where a part is held, loaded, or cooled) stack into `images`. Models that
    need the original columns -- to drop constant conditions, re-normalize them,
    or look one up by name -- read `dataset` instead.

    Attributes:
        tensor: `(n, n_conds)` float tensor on the generator's device, or None
            when the problem has no scalar conditions.
        images: `(n, n_image_conds, H, W)` float tensor of field conditions, or
            None when the problem has none. Native resolution: thermoelastic2d's
            masks are 65x65 against a 64x64 design, because they live on
            finite-element nodes rather than elements.
        dataset: The sampled conditions as an EngiBench/HF dataset, if available.
            The only route to conditions that fit neither tensor.
        keys: Scalar condition names, in the same column order as `tensor`.
        image_keys: Image condition names, in the same channel order as `images`.
    """

    tensor: th.Tensor | None
    images: th.Tensor | None = None
    dataset: Dataset | None = None
    keys: tuple[str, ...] = ()
    image_keys: tuple[str, ...] = ()

    def __len__(self) -> int:
        if self.tensor is not None:
            return int(self.tensor.shape[0])
        if self.images is not None:
            return int(self.images.shape[0])
        return len(self.dataset) if self.dataset is not None else 0

    def require_tensor(self, algo_id: str) -> th.Tensor:
        """Return `tensor`, with a clear error when a conditional model got nothing.

        Raises:
            ValueError: If no scalar conditions are available.
        """
        if self.tensor is None:
            raise ValueError(f"{algo_id} is conditional, but no conditions were provided.")
        return self.tensor

    def require_images(self, algo_id: str) -> th.Tensor:
        """Return `images`, with a clear error when an image-conditioned model got none.

        Raises:
            ValueError: If no image conditions are available.
        """
        if self.images is None:
            raise ValueError(
                f"{algo_id} is image-conditioned, but this problem provides no image conditions. "
                "Declare the problems it serves, or check `image_condition_keys(problem)`."
            )
        return self.images


def pick_device() -> th.device:
    """Return the best available torch device, matching the repo-wide convention."""
    if th.backends.mps.is_available():
        return th.device("mps")
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


INFRA_CONFIG_KEYS = frozenset(
    {
        "algo",
        "problem_id",
        "seed",
        "track",
        "save_model",
        "strict_determinism",
        "checkpoint_backend",
        "sample_interval",
        "wandb_project",
        "wandb_entity",
        "hf_entity",
        "hf_repo_prefix",
        "hf_private",
    }
)
"""Run-config keys that describe *how a run was operated*, not what the model is.

Excluded from the configuration fingerprint so that re-running the same
architecture under a different tracking setup does not look like a new model.
"""


def config_fingerprint(run_config: dict[str, Any] | None) -> str:
    """Short stable hash of the hyperparameters that define a model configuration.

    Two runs of the same algorithm with different `latent_dim` are different
    models and must occupy different leaderboard rows; the same run re-evaluated
    must occupy the same one. `algo_id` and `seed` are not enough to tell those
    apart, so the remaining run-config values are fingerprinted here.

    Returns:
        An 8-character hash, or `"default"` when there is no config to hash.
    """
    if not run_config:
        return "default"
    payload = {k: v for k, v in sorted(run_config.items()) if k not in INFRA_CONFIG_KEYS}
    if not payload:
        return "default"
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:8]


def checkpoint_identity(args: Any) -> dict[str, Any]:
    """Describe a training run's configuration for checkpoint storage.

    Splat into `save_checkpoint_package` so a run is filed both under its exact
    hyperparameters and, when it used the script's defaults, under the canonical
    path that plain `from_pretrained(problem, seed=...)` reads::

        save_checkpoint_package(..., run_config=vars(args), **checkpoint_identity(args))

    Args:
        args: The training script's `Args` dataclass instance.

    Returns:
        `config_fingerprint` and `is_default_config`, ready to pass through.
    """
    config = vars(args)
    try:
        defaults = vars(type(args)())
    except Exception:  # noqa: BLE001 - a non-default-constructible Args is not fatal here
        defaults = {}
    return {
        "config_fingerprint": config_fingerprint(config),
        "is_default_config": bool(defaults) and config_fingerprint(config) == config_fingerprint(defaults),
    }


def config_path_parts(config_fingerprint_value: str | None) -> list[str] | None:
    """Path components addressing one configuration inside a checkpoint repo.

    `None` means the canonical (default-hyperparameter) location.
    """
    if not config_fingerprint_value:
        return None
    return [f"cfg_{config_fingerprint_value}"]


CONDITION_KEYS_METADATA_FIELD = "condition_keys"
"""Checkpoint-metadata field recording which scalar conditions a model was trained on."""

IMAGE_CONDITION_KEYS_METADATA_FIELD = "image_condition_keys"
"""Checkpoint-metadata field recording which image conditions a model was trained on."""


def condition_keys_for(problem: Problem, resolved: ResolvedCheckpoint | None = None) -> tuple[str, ...]:
    """Which conditions a generator consumes, in the order its input tensor holds them.

    A checkpoint records the exact condition schema it was trained under, so a
    network is always rebuilt for the columns it actually saw. Checkpoints
    written before that was recorded fall back to the problem's current scalar
    conditions, which is what those runs used.

    Args:
        problem: The problem the checkpoint was trained on.
        resolved: A downloaded checkpoint package, when one is available.

    Returns:
        Condition names, in `problem.conditions_keys` order.
    """
    if resolved is not None:
        recorded = resolved.metadata.get(CONDITION_KEYS_METADATA_FIELD)
        if recorded:
            return tuple(str(key) for key in recorded)
    return tuple(scalar_condition_keys(problem))


def image_condition_keys_for(problem: Problem, resolved: ResolvedCheckpoint | None = None) -> tuple[str, ...]:
    """Which image conditions a generator consumes, in its input channel order.

    The image counterpart of `condition_keys_for`, with the same precedence: the
    checkpoint's recorded schema wins, so a model keeps loading after a problem
    gains or reorders a field condition.

    Args:
        problem: The problem the checkpoint was trained on.
        resolved: A downloaded checkpoint package, when one is available.

    Returns:
        Condition names, in `problem.conditions_keys` order.
    """
    if resolved is not None:
        recorded = resolved.metadata.get(IMAGE_CONDITION_KEYS_METADATA_FIELD)
        if recorded:
            return tuple(str(key) for key in recorded)
    return tuple(scalar_image_condition_keys(problem))


def design_shape_of(problem: Problem) -> tuple[int, ...]:
    """Shape of a single design for `problem`.

    For `spaces.Dict` problems (e.g. airfoil, whose design is a bundle of named
    arrays) this is the *flattened* shape, since that is the form generators
    emit and metrics consume. Exposed as a function so adapters can size their
    networks before an instance exists.
    """
    space = problem.design_space
    if isinstance(space, spaces.Dict):
        dummy_design, _ = problem.random_design()
        # `spaces.flatten` is typed as returning any flattenable form; for a Dict
        # of Boxes it is always an array, so measure it as one.
        return tuple(np.asarray(spaces.flatten(space, dummy_design)).shape)
    return tuple(space.shape)


class Generator(abc.ABC):
    """Base class for every generative model that can be evaluated or ranked.

    Subclasses declare a few class attributes and implement two methods. See
    `engiopt/generators/_template/adapter.py` for a minimal working example.

    Class attributes:
        algo_id: Stable identifier. Used as the leaderboard key, the HF repo
            suffix, and the registry key. Must match the package directory name.
        conditional: Whether `sample` consumes the scalar conditions.
            Unconditional models still receive them (so the call signature is
            uniform) but ignore them -- which is exactly what the
            conditional-adherence metrics are meant to expose.
        image_conditional: Whether `sample` consumes the *field* conditions --
            thermoelastic2d's masks for where a part is held, loaded, and
            cooled. Declared separately from `conditional` because they are
            different inputs: a model can take the volume budget and ignore the
            boundary conditions. Pointing an image-conditioned model at a
            problem with no field conditions fails immediately, the same way
            `design_kinds` rejects a 2D model on a 3D problem.
        design_kinds: Which design-space shapes this model can serve, e.g.
            `("2d",)`. Used to reject nonsensical model/problem pairings early.
        checkpoint_files: Files the checkpoint package must contain.
        primary_state_key: Key inside the loaded checkpoint holding the
            state dict, e.g. `ckpt["generator"]`.
        output_clip: Optional `(low, high)` clamp applied to sampled designs.
            Several simulators are unstable at exactly 0, so 2D topology models
            clamp to `(1e-3, 1)`. `None` means no clamping. Bounds may be arrays,
            and an instance may override the class default when the range is
            recorded in the checkpoint rather than fixed by the architecture.
    """

    algo_id: ClassVar[str]
    conditional: ClassVar[bool] = True
    image_conditional: ClassVar[bool] = False
    design_kinds: ClassVar[tuple[str, ...]] = ("2d",)
    checkpoint_files: ClassVar[tuple[str, ...]] = ("generator.pth",)
    primary_state_key: ClassVar[str] = "generator"
    output_clip: tuple[Any, Any] | None = None

    def __init__(
        self,
        problem: Problem,
        *,
        problem_id: str,
        seed: int,
        device: th.device,
        run_config: dict[str, Any] | None = None,
        condition_keys: tuple[str, ...] | None = None,
        image_condition_keys: tuple[str, ...] | None = None,
        checkpoint_revision: str | None = None,
        checkpoint_hash: str | None = None,
    ) -> None:
        self.problem = problem
        self.problem_id = problem_id
        self.seed = seed
        self.device = device
        self.run_config: dict[str, Any] = run_config or {}
        self._condition_keys = condition_keys
        self._image_condition_keys = image_condition_keys
        self.checkpoint_revision = checkpoint_revision
        """Commit the checkpoint was downloaded at, when it came from the Hub."""
        self.checkpoint_hash = checkpoint_hash
        """Content hash of the weight files, identifying the exact weights scored."""
        self.last_sample_seconds: float | None = None
        """Wall-clock seconds for the most recent `sample` call (cost metrics)."""

    # ------------------------------------------------------------------
    # The contract
    # ------------------------------------------------------------------

    @classmethod
    @abc.abstractmethod
    def build(
        cls,
        resolved: ResolvedCheckpoint,
        problem: Problem,
        device: th.device,
        **base: Any,
    ) -> Generator:
        """Reconstruct this model from an already-fetched checkpoint package.

        Fetching the package, choosing the device, and selecting the
        configuration are handled by `from_pretrained`; this method only has to
        rebuild the network and hand it to the constructor::

            config = resolved.run_config
            net = MyNet(latent_dim=config["latent_dim"], design_shape=problem.design_space.shape).to(device)
            net.load_state_dict(th.load(resolved.files["generator.pth"], map_location=device)[cls.primary_state_key])
            net.eval()
            return cls(net=net, latent_dim=config["latent_dim"], problem=problem, device=device, **base)

        Args:
            resolved: The downloaded package: `files`, `run_config`, `metadata`.
            problem: The problem this checkpoint was trained on.
            device: Device the model should live on.
            **base: The remaining `Generator.__init__` arguments (`problem_id`,
                `seed`, `run_config`); splat them into `cls(...)` unchanged.
        """

    @classmethod
    def from_pretrained(
        cls,
        problem: Problem,
        *,
        problem_id: str,
        seed: int = 1,
        device: th.device | None = None,
        model_source: ModelSource = "auto",
        hf_entity: str = DEFAULT_HF_ENTITY,
        hf_repo_prefix: str = DEFAULT_HF_REPO_PREFIX,
        local_model_dir: str | None = None,
        config_fingerprint: str | None = None,
        **kwargs: Any,
    ) -> Generator:
        """Rebuild a trained generator from its checkpoint package.

        Args:
            problem: The problem to load a checkpoint for.
            problem_id: Its registry key, which selects the package path.
            seed: Training seed to load.
            device: Torch device; auto-selected when omitted.
            model_source: `auto`, `hf`, or `local`.
            hf_entity: HF org/user holding the checkpoint repos.
            hf_repo_prefix: Prefix of the per-model-family repo.
            local_model_dir: Directory holding an unpacked checkpoint package,
                for `local` (or as the fallback under `auto`).
            config_fingerprint: Load one specific hyperparameter configuration;
                omit for the canonical default-hyperparameter checkpoint.
            **kwargs: Passed through to `build`, for model-specific options.

        Returns:
            The loaded generator, ready to `sample`.
        """
        device = device or pick_device()
        resolved = cls.resolve_checkpoint(
            problem_id=problem_id,
            seed=seed,
            model_source=model_source,
            hf_entity=hf_entity,
            hf_repo_prefix=hf_repo_prefix,
            local_model_dir=local_model_dir,
            config_fingerprint=config_fingerprint,
        )
        return cls.build(
            resolved,
            problem=problem,
            device=device,
            problem_id=problem_id,
            seed=seed,
            run_config=resolved.run_config,
            condition_keys=condition_keys_for(problem, resolved),
            image_condition_keys=image_condition_keys_for(problem, resolved),
            checkpoint_revision=resolved.revision,
            checkpoint_hash=resolved.content_hash,
            **kwargs,
        )

    @abc.abstractmethod
    def _sample(self, conditions: ConditionBatch, n: int) -> th.Tensor | npt.NDArray[Any]:
        """Draw `n` designs. Implement the model-specific sampling procedure here.

        Args:
            conditions: The conditions to satisfy. Use `conditions.tensor` for
                the usual `(n, n_conds)` case, or `conditions.dataset` when the
                model needs the original columns. Reshaping to whatever the
                network expects is the implementation's job.
            n: Number of designs to draw.

        Returns:
            Designs in any shape; `sample` reshapes and clamps them.
        """

    # ------------------------------------------------------------------
    # Shared behaviour
    # ------------------------------------------------------------------

    def sample(
        self,
        conditions: ConditionBatch | th.Tensor | npt.NDArray[Any] | None,
        n: int | None = None,
        seed: int | None = None,
    ) -> npt.NDArray[Any]:
        """Draw designs, timing the call and normalizing the output shape.

        This is the method the evaluator calls. It wraps `_sample` so that every
        model returns `(n, *design_shape)` float arrays and reports its own
        sampling cost, regardless of how it generates internally.

        Args:
            conditions: A `ConditionBatch`, a raw tensor/array of conditions, or
                None. Bare tensors are wrapped for convenience so notebooks can
                call `sample(conditions_tensor)` directly.
            n: Number of designs; inferred from the conditions when omitted.
            seed: Optional torch seed, for reproducible sampling.

        Returns:
            `(n, *design_shape)` array of designs.
        """
        if seed is not None:
            th.manual_seed(seed)
        batch = self._as_batch(conditions)
        if n is None:
            n = len(batch)
            if n == 0:
                raise ValueError("n must be given when no conditions are provided.")

        start = time.perf_counter()
        with th.no_grad():
            raw = self._sample(batch, n)
        # CUDA and MPS queue work asynchronously, so the elapsed time is only
        # the sampling cost once the device has actually finished.
        self._synchronize_device()
        self.last_sample_seconds = time.perf_counter() - start

        designs = raw.detach().cpu().numpy() if isinstance(raw, th.Tensor) else np.asarray(raw)
        designs = designs.reshape(n, *self.design_shape)
        if self.output_clip is not None:
            designs = np.clip(designs, *self.output_clip)
        return designs

    def _synchronize_device(self) -> None:
        """Wait for queued device work, so timings measure sampling rather than dispatch."""
        if self.device.type == "cuda":
            th.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            th.mps.synchronize()

    def _as_batch(self, conditions: ConditionBatch | th.Tensor | npt.NDArray[Any] | None) -> ConditionBatch:
        """Normalize whatever the caller passed into a `ConditionBatch` on this device."""
        if isinstance(conditions, ConditionBatch):
            self._check_condition_keys(conditions.keys, conditions.image_keys)
            return ConditionBatch(
                tensor=self._to_device(conditions.tensor),
                images=self._to_device(conditions.images),
                dataset=conditions.dataset,
                keys=conditions.keys,
                image_keys=conditions.image_keys,
            )
        if conditions is None:
            return ConditionBatch(tensor=None, keys=self.condition_keys)
        tensor = conditions if isinstance(conditions, th.Tensor) else th.as_tensor(np.asarray(conditions))
        return ConditionBatch(tensor=self._to_device(tensor), keys=self.condition_keys)

    def _to_device(self, tensor: th.Tensor | None) -> th.Tensor | None:
        """Move a condition tensor onto this generator's device, passing None through."""
        return None if tensor is None else tensor.to(device=self.device, dtype=th.float)

    def _check_condition_keys(self, keys: tuple[str, ...], image_keys: tuple[str, ...] = ()) -> None:
        """Refuse conditions whose columns are not the ones this model was trained on.

        Silently accepting them produces either a shape error deep inside the
        network or, worse, a design conditioned on the wrong numbers.

        Raises:
            ValueError: If the caller's condition columns differ from the
                checkpoint's, for either the scalar or the image conditions.
        """
        for kind, given, trained in (
            ("conditions", keys, self.condition_keys),
            ("image conditions", image_keys, self.image_condition_keys),
        ):
            if given and trained and tuple(given) != trained:
                raise ValueError(
                    f"{self.algo_id} was trained on {kind} {trained} but was given {tuple(given)}. "
                    "Re-train against the current problem, or evaluate the checkpoint that matches it."
                )

    @cached_property
    def design_shape(self) -> tuple[int, ...]:
        """Shape of a single design; flattened for dict-valued design spaces."""
        return design_shape_of(self.problem)

    @cached_property
    def config_fingerprint(self) -> str:
        """Fingerprint of this checkpoint's hyperparameters; see `config_fingerprint`."""
        return config_fingerprint(self.run_config)

    @cached_property
    def condition_keys(self) -> tuple[str, ...]:
        """Condition columns this model consumes, in input-tensor order.

        Taken from the checkpoint when it recorded them, so a model keeps
        working even if the problem later gains a condition.
        """
        if self._condition_keys is not None:
            return tuple(self._condition_keys)
        return condition_keys_for(self.problem)

    @cached_property
    def image_condition_keys(self) -> tuple[str, ...]:
        """Image condition channels this model consumes, in input-channel order.

        The image counterpart of `condition_keys`, with the same precedence:
        the checkpoint's recorded schema wins over the problem's current one.
        """
        if self._image_condition_keys is not None:
            return tuple(self._image_condition_keys)
        return image_condition_keys_for(self.problem)

    @property
    def n_conds(self) -> int:
        """Number of scalar conditioning variables this model consumes."""
        return len(self.condition_keys)

    @property
    def n_image_conds(self) -> int:
        """Number of image conditioning channels this model consumes."""
        return len(self.image_condition_keys)

    # ------------------------------------------------------------------
    # Checkpoint plumbing shared by every adapter
    # ------------------------------------------------------------------

    @classmethod
    def resolve_checkpoint(
        cls,
        *,
        problem_id: str,
        seed: int,
        model_source: ModelSource = "auto",
        hf_entity: str = DEFAULT_HF_ENTITY,
        hf_repo_prefix: str = DEFAULT_HF_REPO_PREFIX,
        local_model_dir: str | None = None,
        config_fingerprint: str | None = None,
    ) -> ResolvedCheckpoint:
        """Fetch this model's checkpoint package from HuggingFace.

        Args:
            problem_id: EngiBench problem the checkpoint was trained on.
            seed: Training seed to load.
            model_source: Backend preference: `auto`, `hf`, `wandb`, or `local`.
            hf_entity: HF org/user holding the checkpoint repos.
            hf_repo_prefix: Prefix of the per-model-family HF repo.
            local_model_dir: Directory to load from, for `local` sources.
            config_fingerprint: Load one specific hyperparameter configuration.
                Omit it to load the canonical default-hyperparameter checkpoint,
                which is what a bare model name means.
        """
        return resolve_named_checkpoint(
            model_source=model_source,
            problem_id=problem_id,
            algo=cls.algo_id,
            seed=seed,
            hf_entity=hf_entity,
            hf_repo_prefix=hf_repo_prefix,
            required_files=list(cls.checkpoint_files),
            local_model_dir=local_model_dir,
            extra_path_parts=config_path_parts(config_fingerprint),
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(algo_id={self.algo_id!r}, problem_id={self.problem_id!r}, seed={self.seed})"
