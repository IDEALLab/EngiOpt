from dataclasses import dataclass
from email import generator
import os
import tyro
from engibench.utils.all_problems import BUILTIN_PROBLEMS
import torch as th
from torch import nn
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
    track: bool = False
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
    n_epochs: int = 1
    """number of epochs of training"""
    batch_size: int = 32
    """size of the batches"""
    lr: float = 0.0001
    """learning rate"""
    # b1: float = 0.5
    # """decay of first order momentum of gradient"""
    # b2: float = 0.999
    # """decay of first order momentum of gradient"""
    # n_cpu: int = 8
    # """number of cpu threads to use during batch generation"""
    # latent_dim: int = 32
    # """dimensionality of the latent space"""
    # sample_interval: int = 400
    # """interval between image samples"""
    nr_resnet: int = 5
    """Number of residual blocks per stage of the model."""
    nr_filters: int = 160
    """Number of filters to use across the model. Higher = larger model."""
    nr_logistic_mix: int = 10
    """Number of logistic components in the mixture. Higher = more flexible model."""
    resnet_nonlinearity: str = "concat_elu"
    """Nonlinearity to use in the ResNet blocks. One of 'concat_elu', 'elu', 'relu'."""


# IMPLEMENT PIXELCNN++ HERE
class PixelCNNpp(nn.Module):
    def __init__(self):
        super().__init__()

    def discretized_mix_logistic_loss(x, l):
        pass
        # """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
        # xs = int_shape(x) # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
        # ls = int_shape(l) # predicted distribution, e.g. (B,32,32,100)
        # nr_mix = int(ls[-1] / 10) # here and below: unpacking the params of the mixture of logistics
        # logit_probs = l[:,:,:,:nr_mix]
        # l = tf.reshape(l[:,:,:,nr_mix:], xs + [nr_mix*3])
        # means = l[:,:,:,:,:nr_mix]
        # log_scales = tf.maximum(l[:,:,:,:,nr_mix:2*nr_mix], -7.)
        # coeffs = tf.nn.tanh(l[:,:,:,:,2*nr_mix:3*nr_mix])
        # x = tf.reshape(x, xs + [1]) + tf.zeros(xs + [nr_mix]) # here and below: getting the means and adjusting them based on preceding sub-pixels
        # m2 = tf.reshape(means[:,:,:,1,:] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :], [xs[0],xs[1],xs[2],1,nr_mix])
        # m3 = tf.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0],xs[1],xs[2],1,nr_mix])
        # means = tf.concat([tf.reshape(means[:,:,:,0,:], [xs[0],xs[1],xs[2],1,nr_mix]), m2, m3],3)
        # centered_x = x - means
        # inv_stdv = tf.exp(-log_scales)
        # plus_in = inv_stdv * (centered_x + 1./255.)
        # cdf_plus = tf.nn.sigmoid(plus_in)
        # min_in = inv_stdv * (centered_x - 1./255.)
        # cdf_min = tf.nn.sigmoid(min_in)
        # log_cdf_plus = plus_in - tf.nn.softplus(plus_in) # log probability for edge case of 0 (before scaling)
        # log_one_minus_cdf_min = -tf.nn.softplus(min_in) # log probability for edge case of 255 (before scaling)
        # cdf_delta = cdf_plus - cdf_min # probability for all other cases
        # mid_in = inv_stdv * centered_x
        # log_pdf_mid = mid_in - log_scales - 2.*tf.nn.softplus(mid_in) # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

        # # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

        # # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
        # # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

        # # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
        # # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
        # # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
        # # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
        # log_probs = tf.where(x < -0.999, log_cdf_plus, tf.where(x > 0.999, log_one_minus_cdf_min, tf.where(cdf_delta > 1e-5, tf.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(127.5))))

        # log_probs = tf.reduce_sum(log_probs,3) + log_prob_from_logits(logit_probs)
        # if sum_all:
        #     return -tf.reduce_sum(log_sum_exp(log_probs))
        # else:
        #     return -tf.reduce_sum(log_sum_exp(log_probs),[1,2])


if __name__ == "__main__":
    args = tyro.cli(Args)

    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    design_shape = problem.design_space.shape
    print(f"Design shape: {design_shape}")
    conditions = problem.conditions
    nr_conditions = len(conditions)


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
    loss = PixelCNNpp.discretized_mix_logistic_loss

    # Initialize model
    # ... implement

    # model.to(device)
    # loss.to(device)

    # Configure data loader
    training_ds = problem.dataset.with_format("torch", device=device)["train"]
    condition_tensors = [training_ds[key][:] for key in problem.conditions_keys]

    training_ds = th.utils.data.TensorDataset(training_ds["optimal_design"][:].flatten(1), *condition_tensors)

    dataloader = th.utils.data.DataLoader(
        training_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Optimzer
    # optimizer = th.optim.Adam(model.parameters(), lr=args.lr) # add other args if necessary

    @th.no_grad()
    def sample_designs(n_designs: int) -> tuple[th.Tensor, th.Tensor]:
        """Sample designs from trained model."""
        # Is that needed?
        # z = th.randn((n_designs, args.latent_dim), device=device, dtype=th.float)

        # Create condition grid
        all_conditions = th.stack(condition_tensors, dim=1)
        linspaces = [
            th.linspace(all_conditions[:, i].min(), all_conditions[:, i].max(), n_designs, device=device)
            for i in range(all_conditions.shape[1])
        ]
        desired_conds = th.stack(linspaces, dim=1)

        # implement sampling from model here
        gen_imgs = None

        return desired_conds, gen_imgs



    # ----------
    #  Training
    # ----------
    for epoch in tqdm.trange(args.n_epochs):
        for i, data in enumerate(dataloader):
            print(data[0])
            print(data[1:])
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