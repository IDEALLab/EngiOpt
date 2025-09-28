"""based on the diffuser example from huggingface: https://huggingface.co/learn/diffusion-course/unit1/2 ."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import random
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

if TYPE_CHECKING:
    from collections.abc import Callable


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
    seed: int = 1
    """Random seed."""
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

    num_timesteps: int = 250
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
        device: th.device = th.device("cpu"),  # noqa: B008
    ) -> th.Tensor:
        """Takes an image, noise and step; returns denoised image."""
        betas_t = get_index_from_list(self.betas, t, x_noisy.shape).to(device)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_noisy.shape).to(
            device
        )
        sqrt_recip_alphas_t = get_index_from_list(self.sqrt_recip_alphas, t, x_noisy.shape).to(device)
        model_mean = sqrt_recip_alphas_t * (x_noisy - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)
        posterior_variance_t = get_index_from_list(self.posterior_variance, t, x_noisy.shape).to(device)

        # mean + variance
        return (model_mean + th.sqrt(posterior_variance_t) * noise_pred).to(device)

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
            betas_t = get_index_from_list(self.betas, t, x.shape)
            sqrt_one_minus_alphas_cumprod_t = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            sqrt_recip_alphas_t = get_index_from_list(self.sqrt_recip_alphas, t, x.shape)

            # Call model (current image - noise prediction)
            noise_pred = model(x, t, encoder_hidden_states).sample
            model_mean = sqrt_recip_alphas_t * (x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)

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
    th.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    th.backends.cudnn.deterministic = True

    os.makedirs("images", exist_ok=True)

    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    # Loss function
    adversarial_loss: th.nn.Module = th.nn.MSELoss()
    encoder_hid_dim = len(problem.conditions)
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
    filtered_ds_max = filtered_ds.max()
    filtered_ds_min = filtered_ds.min()
    filtered_ds_norm = (filtered_ds - filtered_ds_min) / (filtered_ds_max - filtered_ds_min)
    training_ds = th.utils.data.TensorDataset(
        filtered_ds_norm.flatten(1), *[training_ds[key][:] for key in problem.conditions_keys]
    )
    cond_tensors = th.stack(training_ds.tensors[1 : len(problem.conditions) + 1])
    conds_min = cond_tensors.amin(dim=tuple(range(1, cond_tensors.ndim)))
    conds_max = cond_tensors.amax(dim=tuple(range(1, cond_tensors.ndim)))

    dataloader = th.utils.data.DataLoader(
        training_ds,
        batch_size=args.batch_size,
        shuffle=True,
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
                        img = tensor.cpu().numpy()  # Extract x and y coordinates
                        dc = hidden_states[j, 0, :].cpu()
                        axes[j].imshow(img[0])  # image plot
                        title = [(problem.conditions[i][0], f"{dc[i]:.2f}") for i in range(len(problem.conditions))]
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
                    }

                    th.save(ckpt_model, "model.pth")
                    if args.track:
                        artifact_model = wandb.Artifact(f"{args.problem_id}_{args.algo}_model", type="model")
                        artifact_model.add_file("model.pth")

                        wandb.log_artifact(artifact_model, aliases=[f"seed_{args.seed}"])

    wandb.finish()
