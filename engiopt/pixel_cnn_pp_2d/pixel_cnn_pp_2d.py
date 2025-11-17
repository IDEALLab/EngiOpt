from dataclasses import dataclass
import os
import tyro
from engibench.utils.all_problems import BUILTIN_PROBLEMS
import torch as th
import numpy as np
import random
import tqdm
import time
import matplotlib.pyplot as plt

import wandb


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

    # CHANGE!
    # Algorithm specific
    n_epochs: int = 200
    """number of epochs of training"""
    batch_size: int = 32
    """size of the batches"""
    lr: float = 0.0001
    """learning rate"""
    b1: float = 0.5
    """decay of first order momentum of gradient"""
    b2: float = 0.999
    """decay of first order momentum of gradient"""
    n_cpu: int = 8
    """number of cpu threads to use during batch generation"""
    latent_dim: int = 32
    """dimensionality of the latent space"""
    sample_interval: int = 400
    """interval between image samples"""


# IMPLEMENT PIXELCNN++ HERE



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
    # ... implement

    # Initialize model
    # ... implement

    # model.to(device)
    # loss.to(device)

    # Configure data loader
    training_ds = problem.dataset.with_format("torch", device=device)["train"]
    # ...
    
    dataloader = th.utils.data.DataLoader(
        training_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Training loop
    # optimizer = th.optim.Adam(model.parameters(), lr=args.lr) # add other args if necessary

    # @th.no_grad()
    # def sample_designs(model: ---, n_designs: int = 25) -> tuple[th.Tensor, th.Tensor]:
        # ... implement


    
    # ----------
    #  Training
    # ----------
    for epoch in tqdm.trange(args.n_epochs):
        for i, data in enumerate(dataloader):
            batch_start_time = time.time()
            # ... implement

            # Backpropagation
            # loss.backward()
            # optimizer.step()


            # ----------
            #  Logging
            # ----------
            if args.track:
                batches_done = epoch * len(dataloader) + i
                wandb.log(
                    {
                        "loss": None,  #loss.item(),
                        "epoch": epoch,
                        "batch": batches_done,
                    }
                )
                print(
                    f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(dataloader)}] [loss: {None}]] [{time.time() - batch_start_time:.2f} sec]" #loss.item()
                )

                # This saves a grid image of 25 generated designs every sample_interval
                if batches_done % args.sample_interval == 0:
                    # Extract 25 designs

                    designs, hidden_states = None #sample_designs(model, 25)
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
                        "model": None, # model.state_dict(),
                        "optimizer_generator": None, # optimizer.state_dict(),
                        "loss": None,  # loss.item(),
                    }

                    th.save(ckpt_model, "model.pth")
                    if args.track:
                        artifact_model = wandb.Artifact(f"{args.problem_id}_{args.algo}_model", type="model")
                        artifact_model.add_file("model.pth")

                        wandb.log_artifact(artifact_model, aliases=[f"seed_{args.seed}"])

    wandb.finish()