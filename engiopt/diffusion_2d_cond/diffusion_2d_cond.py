"""based on the diffuser example from huggingface: https://huggingface.co/learn/diffusion-course/unit1/2 ."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import time
from typing import Literal, TYPE_CHECKING

from diffusers import UNet2DConditionModel
from engibench.utils.all_problems import BUILTIN_PROBLEMS
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch.nn import functional
import tqdm
import tyro
import wandb

from engiopt.checkpoint_store import save_checkpoint_package
from engiopt.reproducibility import enable_strict_determinism
from engiopt.reproducibility import make_dataloader_generator
from engiopt.reproducibility import seed_training

if TYPE_CHECKING:
    from collections.abc import Callable

DIFFUSION_SAMPLE_MIN = -1.0
DIFFUSION_SAMPLE_MAX = 1.0


@dataclass
class Args:
    """Command-line arguments."""

    problem_id: str = "beams2d"
    """Problem identifier."""
    algo: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this algorithm."""

    # Tracking
    track: bool = True
    """Track the experiment with wandb."""
    wandb_project: str = "engiopt"
    """Wandb project name."""
    wandb_entity: str | None = None
    """Wandb entity name."""
    hf_entity: str = "IDEALLab"
    """HF org/user where checkpoints are stored."""
    hf_repo_prefix: str = "engiopt"
    """HF repo prefix used for model-family repositories."""
    seed: int = 1
    """Random seed."""

    strict_determinism: bool = False
    """Enable strict deterministic operations for reproducibility debugging."""
    save_model: bool = False
    """Saves the model to disk."""

    # Algorithm specific
    n_epochs: int = 200
    """number of epochs of training"""
    batch_size: int = 32
    """size of the batches"""
    lr: float = 4e-4
    """learning rate"""
    b1: float = 0.5
    """decay of first order momentum of gradient"""
    b2: float = 0.999
    """decay of first order momentum of gradient"""
    n_cpu: int = 8
    """number of cpu threads to use during batch generation"""
    latent_dim: int = 100
    """dimensionality of the latent space"""
    sample_interval: int = 400
    """interval between image samples"""

    num_timesteps: int = 1000
    """Number of timesteps in the diffusion schedule"""
    layers_per_block: int = 2
    """Layers per U-NET block"""
    noise_schedule: Literal["linear", "cosine", "exp"] = "linear"
    """Diffusion schedule ('linear', 'cosine', 'exp')"""


def beta_schedule(
    t: int, start: float = 1e-4, end: float = 0.02, scale: float = 1.0, options: dict | None = None
) -> th.Tensor:
    """Returns a beta schedule (default: linear) for the diffusion model.

    Args:
        t: Number of timesteps
        start: Starting value of beta
        end: Ending value of beta
        scale: Scaling factor for beta
        options: Dictionary containing optional parameters (cosine, exp_biasing, exp_bias_factor)
        cosine: Whether to use a cosine beta schedule
        exp_biasing: Whether to use exponential biasing
        exp_bias_factor: Exponential biasing factor
    """
    beta: th.Tensor = th.linspace(scale * start, scale * end, t)
    cosine = options.get("cosine", False) if options else False
    exp_biasing = options.get("exp_biasing", False) if options else False
    exp_bias_factor = options.get("exp_bias_factor", 1) if options else 1

    if cosine:
        beta_list: list[float] = []

        def a_func(t_val: float) -> float:
            return math.cos((t_val + 0.008) / 1.008 * np.pi / 2) ** 2

        for i in range(t):
            t1 = i / t
            t2 = (i + 1) / t
            beta_list.append(min(1 - a_func(t2) / a_func(t1), 0.999))

        beta = th.tensor(beta_list)

    if exp_biasing:
        beta = (th.flip(th.exp(-exp_bias_factor * th.linspace(0, 1, t)), dims=[0])) * beta

    return beta


def get_index_from_list(vals: th.Tensor, t: th.Tensor, x_shape: tuple[int, ...]) -> th.Tensor:
    """Returns a specific index t of a passed list of values vals.

    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def normalize_designs_to_diffusion_range(
    designs: th.Tensor,
    design_min: th.Tensor,
    design_max: th.Tensor,
) -> th.Tensor:
    """Map designs from their problem bounds to [-1, 1]."""
    denom = th.clamp(design_max - design_min, min=th.finfo(designs.dtype).eps)
    designs_01 = (designs - design_min) / denom
    return designs_01 * (DIFFUSION_SAMPLE_MAX - DIFFUSION_SAMPLE_MIN) + DIFFUSION_SAMPLE_MIN


def denormalize_designs_from_diffusion_range(
    designs: th.Tensor,
    design_min: th.Tensor,
    design_max: th.Tensor,
) -> th.Tensor:
    """Map designs from [-1, 1] back to the original problem scale."""
    designs_01 = (designs - DIFFUSION_SAMPLE_MIN) / (DIFFUSION_SAMPLE_MAX - DIFFUSION_SAMPLE_MIN)
    return designs_01 * (design_max - design_min) + design_min


def get_design_bounds(
    problem,
    fallback_designs: th.Tensor,
    device: th.device,
) -> tuple[th.Tensor, th.Tensor]:
    """Get finite design bounds from the EngiBench problem definition."""
    design_min = th.as_tensor(problem.design_space.low, dtype=fallback_designs.dtype, device=device)
    design_max = th.as_tensor(problem.design_space.high, dtype=fallback_designs.dtype, device=device)
    if th.isfinite(design_min).all() and th.isfinite(design_max).all() and th.all(design_max > design_min):
        return design_min, design_max

    return fallback_designs.min(), fallback_designs.max()


class DiffusionSampler:
    # Precompute the sqrt alphas and sqrt one minus alphas
    def __init__(self, t: int, betas: th.Tensor):
        self.t = t
        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = th.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = th.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = th.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = th.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef1 = self.betas * th.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * th.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    def _posterior_mean(self, noise_pred: th.Tensor, x_noisy: th.Tensor, t: th.Tensor) -> th.Tensor:
        sqrt_alphas_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, x_noisy.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod,
            t,
            x_noisy.shape,
        )
        pred_original_sample = (x_noisy - sqrt_one_minus_alphas_cumprod_t * noise_pred) / sqrt_alphas_cumprod_t
        pred_original_sample = pred_original_sample.clamp(DIFFUSION_SAMPLE_MIN, DIFFUSION_SAMPLE_MAX)

        posterior_mean_coef1_t = get_index_from_list(self.posterior_mean_coef1, t, x_noisy.shape)
        posterior_mean_coef2_t = get_index_from_list(self.posterior_mean_coef2, t, x_noisy.shape)
        return posterior_mean_coef1_t * pred_original_sample + posterior_mean_coef2_t * x_noisy

    def forward_diffusion_sample(
        self,
        x_0: th.Tensor,
        t: th.Tensor,
        device: th.device = th.device("cpu"),  # noqa: B008
    ) -> tuple[th.Tensor, th.Tensor]:
        """Takes an image and a timestep as input and returns the noisy version of it.

        Returns the noisy version of the input image at the specified timestep.
        """
        noise = th.randn_like(x_0).to(device)
        sqrt_alphas_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        # mean + variance
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(
            device
        ), noise.to(device)

    def forward_diffusion_sample_partial(
        self,
        x_0: th.Tensor,
        t_current: th.Tensor,
        t_final: th.Tensor,
        device: th.device = th.device("cpu"),  # noqa: B008
    ) -> tuple[th.Tensor, th.Tensor]:
        """Takes an image at a timestep and.

        adds noise to reach the desired timestep.
        """
        for i in range(t_final[0] - t_current[0]):
            t = t_final - i

            noise = th.randn_like(
                x_0,
            ).to(device)
            x_0 = th.sqrt(get_index_from_list(self.alphas, t, x_0.shape)) * x_0.to(device) + th.sqrt(
                get_index_from_list(1 - self.alphas, t, x_0.shape)
            ) * noise.to(device)

        # mean + variance
        return x_0, noise.to(device)

    def diffusion_step_sample(
        self,
        noise_pred: th.Tensor,
        x_noisy: th.Tensor,
        t: th.Tensor,
        device: th.device | None = None,
    ) -> th.Tensor:
        """Takes an image, noise and step; returns denoised image."""
        if device is None:
            device = x_noisy.device

        model_mean = self._posterior_mean(noise_pred, x_noisy, t).to(device)
        posterior_variance_t = get_index_from_list(self.posterior_variance, t, x_noisy.shape).to(device)
        t_mask = ((t != 0).float().view(-1, *([1] * (len(x_noisy.shape) - 1)))).to(device)

        # mean + variance
        return (model_mean + th.sqrt(posterior_variance_t) * th.randn_like(x_noisy) * t_mask).to(device)

    def lossfn_builder(self) -> Callable[[th.Tensor, th.Tensor], th.Tensor]:
        """Returns the loss function for the diffusion model."""

        def lossfn(noise_pred: th.Tensor, noise: th.Tensor) -> th.Tensor:
            return functional.mse_loss(noise_pred, noise)

        return lossfn

    def sample_timestep(
        self,
        model: UNet2DConditionModel,
        x: th.Tensor,
        t: th.Tensor,
        encoder_hidden_states: th.Tensor,
        t_mask: th.Tensor | None = None,
    ) -> th.Tensor:
        """Calls the model to predict the noise in the image and returns the denoised image.

        Applies noise to this image, if we are not in the last step yet.
        """
        model.eval()
        with th.no_grad():
            # Call model (current image - noise prediction)
            noise_pred = model(x, t, encoder_hidden_states).sample
            model_mean = self._posterior_mean(noise_pred, x, t)

            posterior_variance_t = get_index_from_list(self.posterior_variance, t, x.shape)
            if t_mask is None:
                device = x.device

                t_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))).to(device)

            return model_mean + th.sqrt(posterior_variance_t) * th.randn_like(x) * t_mask


if __name__ == "__main__":
    args = tyro.cli(Args)

    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    design_shape = problem.design_space.shape

    # Logging
    run_name = f"{args.problem_id}__{args.algo}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), save_code=True, name=run_name)

    # Seeding
    rng = seed_training(args.seed)
    if args.strict_determinism:
        enable_strict_determinism(warn_only=True)

    os.makedirs("images", exist_ok=True)

    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    # Loss function
    adversarial_loss: th.nn.Module = th.nn.MSELoss()
    encoder_hid_dim = len(problem.conditions_keys)
    # Initialize UNet from Huggingface
    model = UNet2DConditionModel(
        sample_size=design_shape,
        in_channels=1,
        out_channels=1,
        cross_attention_dim=64,
        block_out_channels=(32, 64, 128, 256),
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        layers_per_block=args.layers_per_block,
        transformer_layers_per_block=1,
        encoder_hid_dim=encoder_hid_dim,
        only_cross_attention=True,
    )

    model.to(device)
    adversarial_loss.to(device)

    # Configure data loader
    training_ds = problem.dataset.with_format("torch", device=device)["train"]
    filtered_ds = th.zeros(len(training_ds), design_shape[0], design_shape[1], device=device)
    for i in range(len(training_ds)):
        filtered_ds[i] = training_ds[i]["optimal_design"][:].reshape(1, design_shape[0], design_shape[1])
    filtered_ds_min, filtered_ds_max = get_design_bounds(problem, filtered_ds, device)
    filtered_ds_norm = normalize_designs_to_diffusion_range(filtered_ds, filtered_ds_min, filtered_ds_max)
    training_ds = th.utils.data.TensorDataset(
        filtered_ds_norm.flatten(1), *[training_ds[key][:] for key in problem.conditions_keys]
    )
    cond_tensors = th.stack(training_ds.tensors[1 : len(problem.conditions_keys) + 1])
    conds_min = cond_tensors.amin(dim=tuple(range(1, cond_tensors.ndim)))
    conds_max = cond_tensors.amax(dim=tuple(range(1, cond_tensors.ndim)))

    dataloader = th.utils.data.DataLoader(
        training_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=make_dataloader_generator(args.seed),
    )
    num_timesteps = args.num_timesteps

    # Training loop
    optimizer = th.optim.AdamW(model.parameters(), lr=args.lr)

    ## Schedule Parameters
    start = 1e-4  # Starting variance
    end = 0.02  # Ending variance

    # Choose a schedule (if the following are False, then a linear schedule is used)

    options = {
        "cosine": args.noise_schedule == "cosine",  # Use cosine schedule
        "exp_biasing": args.noise_schedule == "exp",  # Use exponential schedule
        "exp_bias_factor": 1,  # Exponential schedule factor (used if exp_biasing=True)
    }
    ##

    # Choose a variance schedule
    betas = beta_schedule(t=num_timesteps, start=start, end=end, scale=1.0, options=options)

    ddm_sampler = DiffusionSampler(num_timesteps, betas)

    # Loss function
    def ddm_loss_fn(noise_pred: th.Tensor, noise: th.Tensor) -> th.Tensor:
        """Compute the MSE loss between predicted and target noise.

        Args:
            noise_pred: The predicted noise tensor
            noise: The target noise tensor

        Returns:
            The computed MSE loss between predictions and targets
        """
        return functional.mse_loss(noise_pred, noise)

    @th.no_grad()
    def sample_designs(model: UNet2DConditionModel, n_designs: int = 25) -> tuple[th.Tensor, th.Tensor]:
        """Samples n_designs designs."""
        model.eval()
        with th.no_grad():
            dims = (n_designs, 1, design_shape[0], design_shape[1])
            steps = th.linspace(0, 1, n_designs, device=device).view(n_designs, 1, 1)
            encoder_hidden_states = conds_min + steps * (conds_max - conds_min)
            image = th.randn(dims, device=device)  # initial image
            for i in range(num_timesteps)[::-1]:
                t = th.full((n_designs,), i, device=device, dtype=th.long)

                image = ddm_sampler.sample_timestep(model, image, t, encoder_hidden_states)

        return image, encoder_hidden_states

    # ----------
    #  Training
    # ----------
    for epoch in tqdm.trange(args.n_epochs):
        for i, data in enumerate(dataloader):
            batch_start_time = time.time()
            # Zero the parameter gradients
            optimizer.zero_grad()
            designs = data[0].reshape(-1, 1, design_shape[0], design_shape[1])
            x = designs.to(device)
            conds = th.stack((data[1:]), dim=1).reshape(-1, 1, encoder_hid_dim)

            current_batch_size = x.shape[0]
            t = th.randint(0, num_timesteps, (current_batch_size,), device=device).long()
            encoder_hidden_states = conds.to(device)

            # Get the noise and the noisy input
            x_noisy, noise = ddm_sampler.forward_diffusion_sample(x, t, device)

            noise_pred = model(x_noisy, t, encoder_hidden_states).sample
            loss = ddm_loss_fn(noise_pred, noise)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # ----------
            #  Logging
            # ----------
            if args.track:
                batches_done = epoch * len(dataloader) + i
                wandb.log(
                    {
                        "loss": loss.item(),
                        "epoch": epoch,
                        "batch": batches_done,
                    }
                )
                print(
                    f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(dataloader)}] [loss: {loss.item()}]] [{time.time() - batch_start_time:.2f} sec]"
                )

                # This saves a grid image of 25 generated designs every sample_interval
                if batches_done % args.sample_interval == 0:
                    # Extract 25 designs

                    designs, hidden_states = sample_designs(model, 25)
                    fig, axes = plt.subplots(5, 5, figsize=(12, 12))

                    # Flatten axes for easy indexing
                    axes = axes.flatten()

                    # Plot the image created by each output
                    for j, tensor in enumerate(designs):
                        design = denormalize_designs_from_diffusion_range(tensor, filtered_ds_min, filtered_ds_max)
                        img = design.cpu().numpy()  # Extract x and y coordinates
                        dc = hidden_states[j, 0, :].cpu()
                        axes[j].imshow(img[0])  # image plot
                        title = [(problem.conditions_keys[i], f"{dc[i]:.2f}") for i in range(len(problem.conditions_keys))]
                        title_string = "\n ".join(f"{condition}: {value}" for condition, value in title)
                        axes[j].title.set_text(title_string)  # Set title
                        axes[j].set_xticks([])  # Hide x ticks
                        axes[j].set_yticks([])  # Hide y ticks

                    plt.tight_layout()
                    img_fname = f"images/{batches_done}.png"
                    plt.savefig(img_fname)
                    plt.close()
                    wandb.log({"designs": wandb.Image(img_fname)})

                # --------------
                #  Save models
                # --------------
                if args.save_model and epoch == args.n_epochs - 1 and i == len(dataloader) - 1:
                    ckpt_model = {
                        "epoch": epoch,
                        "batches_done": batches_done,
                        "model": model.state_dict(),
                        "optimizer_generator": optimizer.state_dict(),
                        "loss": loss.item(),
                        "design_min": filtered_ds_min.detach().cpu(),
                        "design_max": filtered_ds_max.detach().cpu(),
                        "diffusion_sample_min": DIFFUSION_SAMPLE_MIN,
                        "diffusion_sample_max": DIFFUSION_SAMPLE_MAX,
                    }

                    th.save(ckpt_model, "model.pth")
                    save_checkpoint_package(
                        checkpoint_backend="hf",
                        hf_entity=args.hf_entity,
                        hf_repo_prefix=args.hf_repo_prefix,
                        hf_private=False,
                        problem_id=args.problem_id,
                        algo=args.algo,
                        seed=args.seed,
                        checkpoint_files={"model.pth": "model.pth"},
                        run_config=vars(args),
                        primary_files=["model.pth"],
                    )

    wandb.finish()
