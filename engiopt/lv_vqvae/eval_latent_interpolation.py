"""Standalone Evaluation Script: Latent Space Interpolation for LV-VQVAE / LVAE.

Loads a pretrained Stage 1 checkpoint from Weights & Biases and interpolates
between MAXIMALLY DIVERSE pairs of valid designs in the latent space to check
for manifold smoothness vs. spaghetti curves.
"""

from __future__ import annotations

import dataclasses
import os
import random

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import tyro
import wandb

from engiopt.lv_vqvae.lv_vqvae import VQVAE
from engiopt.transforms import resize_to


@dataclasses.dataclass
class Args:
    problem_id: str = "beams2d"
    """Problem identifier."""
    seed: int = 1
    """Random seed."""
    wandb_project: str = "engiopt"
    """Wandb project name."""
    wandb_entity: str | None = None
    """Wandb entity name."""
    n_interpolations: int = 4
    """Number of design pairs to interpolate between (rows in the plot)."""
    n_steps: int = 10
    """Number of interpolation steps between Design A and Design B (columns)."""


if __name__ == "__main__":
    args = tyro.cli(Args)
    seed = args.seed

    # Setup Device & RNG
    th.manual_seed(seed)
    random.seed(seed)
    np.random.Generator(np.random.PCG64(seed))
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    print(f"Loading problem: {args.problem_id}")
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=seed)
    design_shape = problem.design_space.shape

    # --------------------------------------------------------------------------
    # W&B Artifact Retrieval
    # --------------------------------------------------------------------------
    print("Fetching Stage 1 model from Weights & Biases...")
    if args.wandb_entity is not None:
        artifact_path_vqvae = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_lv_vqvae_lv_vqvae:seed_{seed}"
    else:
        artifact_path_vqvae = f"{args.wandb_project}/{args.problem_id}_lv_vqvae_lv_vqvae:seed_{seed}"

    api = wandb.Api()
    artifact_vqvae = api.artifact(artifact_path_vqvae, type="model")

    class RunRetrievalError(ValueError):
        def __init__(self):
            super().__init__("Failed to retrieve the run")

    run = artifact_vqvae.logged_by()
    if run is None:
        raise RunRetrievalError
    run = api.run(f"{run.entity}/{run.project}/{run.id}")

    artifact_dir_vqvae = artifact_vqvae.download()
    ckpt_path_vqvae = os.path.join(artifact_dir_vqvae, "lv_vqvae.pth")
    ckpt = th.load(ckpt_path_vqvae, map_location=device, weights_only=False)

    # --------------------------------------------------------------------------
    # Prepare Validation Data
    # --------------------------------------------------------------------------
    val_ds = problem.dataset.with_format("torch")["val"]
    val_ds = val_ds.map(
        lambda batch: {
            "optimal_upsampled": resize_to(
                data=batch["optimal_design"][:],
                h=run.config["image_size"],
                w=run.config["image_size"]
            ).cpu().numpy()
        },
        batched=True,
    )

    # We only need the designs for Stage 1 interpolation, not the conditions
    th_val_ds = th.utils.data.TensorDataset(th.as_tensor(val_ds["optimal_upsampled"][:]).to(device))

    # --------------------------------------------------------------------------
    # Initialize Continuous LVAE
    # --------------------------------------------------------------------------
    print("Initializing continuous LVAE...")
    vqvae = VQVAE(
        device=device,
        is_c=False,
        use_vq=False,  # Force continuous mode
        encoder_channels=run.config["encoder_channels"],
        encoder_start_resolution=run.config["image_size"],
        encoder_attn_resolutions=run.config["encoder_attn_resolutions"],
        encoder_num_res_blocks=run.config["encoder_num_res_blocks"],
        decoder_channels=run.config["decoder_channels"],
        decoder_start_resolution=run.config["latent_size"],
        decoder_num_res_blocks=run.config["decoder_num_res_blocks"],
        image_channels=run.config["image_channels"],
        latent_dim=run.config["latent_dim"],
        num_codebook_vectors=run.config["num_codebook_vectors"],
    ).to(device)

    vqvae.load_state_dict(ckpt["vqvae"])
    vqvae.eval()

    # Extract Pruning State
    lv_state = ckpt.get("lv_state", {})
    active_mask = lv_state.get("active_mask", th.ones(run.config["latent_dim"], dtype=th.bool)).to(device)
    frozen_mean = lv_state.get("frozen_mean", th.zeros(run.config["latent_dim"])).to(device)

    active_dims = int(active_mask.sum().item())
    print(f"Model loaded. Active dimensions: {active_dims} / {run.config['latent_dim']}")

    # --------------------------------------------------------------------------
    # Diverse Pair Mining (Farthest Point Sampling)
    # --------------------------------------------------------------------------
    print("Mining validation set for 8 mutually diverse designs using Farthest Point Sampling...")

    # Load a large batch to search through (256 designs)
    mining_loader = th.utils.data.DataLoader(th_val_ds, batch_size=256, shuffle=True)
    candidate_designs = next(iter(mining_loader))[0].to(dtype=th.float32, device=device)

    # Flatten designs to compute structural distance
    flat_designs = candidate_designs.view(candidate_designs.shape[0], -1)

    # Compute pairwise Euclidean distance matrix
    pairwise_dists = th.cdist(flat_designs, flat_designs, p=2)

    # 1. Farthest Point Sampling (FPS) to find N*2 mutually distinct designs
    n_total_designs = args.n_interpolations * 2
    selected_indices = []

    # Start with the absolute furthest pair in the whole batch
    max_idx = th.argmax(pairwise_dists)
    first_idx = max_idx // pairwise_dists.shape[1]
    second_idx = max_idx % pairwise_dists.shape[1]
    selected_indices.extend([first_idx.item(), second_idx.item()])

    # Iteratively add the point that is furthest from the already selected points
    for _ in range(n_total_designs - 2):
        # Distances from all candidate points to ONLY the currently selected points
        dists_to_selected = pairwise_dists[:, selected_indices]

        # For each candidate, find the distance to its CLOSEST selected point
        min_dists, _ = th.min(dists_to_selected, dim=1)

        # Prevent re-selecting already chosen points
        min_dists[selected_indices] = -1.0

        # Pick the candidate where this minimum distance is MAXIMUM
        next_idx = th.argmax(min_dists).item()
        selected_indices.append(next_idx)

    # 2. Greedy Pairing (Pair them up to maximize interpolation distance)
    idx_a = []
    idx_b = []
    unpaired = list(selected_indices)

    while len(unpaired) > 0:
        if len(unpaired) == 2:  # noqa: PLR2004
            idx_a.append(unpaired[0])
            idx_b.append(unpaired[1])
            break

        # Extract the sub-matrix of distances for ONLY the remaining unpaired indices
        sub_dists = pairwise_dists[unpaired][:, unpaired]

        # Mask out self-pairs and duplicates to find the max distance pair
        mask = th.triu(th.ones_like(sub_dists, dtype=th.bool), diagonal=1)
        sub_dists[~mask] = -1.0

        max_sub_idx = th.argmax(sub_dists)
        u_idx1 = max_sub_idx // sub_dists.shape[1]
        u_idx2 = max_sub_idx % sub_dists.shape[1]

        # Add to pair lists
        val1, val2 = unpaired[u_idx1], unpaired[u_idx2]
        idx_a.append(val1)
        idx_b.append(val2)

        # Remove them from the pool
        unpaired.remove(val1)
        unpaired.remove(val2)

    print(f"Successfully selected and paired {len(idx_a) * 2} unique designs.")

    # --------------------------------------------------------------------------
    # Run Interpolations
    # --------------------------------------------------------------------------
    print(f"Generating {args.n_interpolations} interpolations with {args.n_steps} steps each...")

    fig, axes = plt.subplots(args.n_interpolations, args.n_steps, figsize=(2 * args.n_steps, 2.5 * args.n_interpolations))
    if args.n_interpolations == 1:
        axes = [axes]

    with th.no_grad():
        for i in range(args.n_interpolations):
            # Extract the maximally diverse pair
            design_a = candidate_designs[idx_a[i]].unsqueeze(0)
            design_b = candidate_designs[idx_b[i]].unsqueeze(0)

            # Encode to continuous latent space
            z1, *_ = vqvae.encode(design_a, active_mask=active_mask, frozen_mean=frozen_mean)
            z2, *_ = vqvae.encode(design_b, active_mask=active_mask, frozen_mean=frozen_mean)

            # Create interpolation steps
            alphas = th.linspace(0, 1, steps=args.n_steps, device=device).view(-1, 1, 1, 1)

            # Linear interpolation in latent space
            z_interp = z1 * (1 - alphas) + z2 * alphas

            # Decode the entire path
            decoded_interps = vqvae.decode(z_interp, active_mask=active_mask, frozen_mean=frozen_mean)

            # Plot this row
            for j in range(args.n_steps):
                ax = axes[i][j]
                img = decoded_interps[j].cpu().numpy().squeeze()

                # Clip to [0,1] for visualization
                img = np.clip(img, 0, 1)

                ax.imshow(img, cmap="gray_r", vmin=0, vmax=1)
                ax.axis("off")

                if j == 0:
                    ax.set_title(f"Pair {i+1}:\nDesign A", fontsize=10)
                elif j == args.n_steps - 1:
                    ax.set_title(f"Pair {i+1}:\nDesign B", fontsize=10)
                elif i == 0:
                    ax.set_title(f"Step {j}", fontsize=10)

    plt.tight_layout()

    # Save Output
    os.makedirs("evals", exist_ok=True)
    out_path = f"evals/interpolation_{args.problem_id}_seed{seed}.png"
    plt.savefig(out_path, dpi=500, bbox_inches="tight")
    plt.close()

    print(f"Saved interpolation grid to {out_path}")
