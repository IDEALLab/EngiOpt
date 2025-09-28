"""A pytorch implementation of the Bezier GAN from https://github.com/IDEALLab/bezier-gan .

See https://arxiv.org/abs/1808.08871 for more details on this algorithm.
Conditional bezier GAN based on https://github.com/IDEALLab/CEBGAN_JMD_2021 , http://dx.doi.org/10.1115/1.4052846.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import random
import time
from typing import TYPE_CHECKING

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as f
import tyro

import wandb

if TYPE_CHECKING:
    from collections.abc import Callable

_EPS = 1e-7


@dataclass
class Args:
    """Command-line arguments."""

    problem_id: str = "airfoil"
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
    seed: int = 6
    """Random seed."""
    save_model: bool = True
    """Saves the model to disk."""

    # Algorithm specific
    n_epochs: int = 5000
    """number of epochs of training"""
    batch_size: int = 32
    """size of the batches"""
    lr_gen: float = 0.00005
    """learning rate for the generator"""
    lr_disc: float = 0.0002
    """learning rate for the discriminator"""
    b1: float = 0.5
    """decay of first order momentum of gradient"""
    b2: float = 0.999
    """decay of first order momentum of gradient"""
    n_cpu: int = 8
    """number of cpu threads to use during batch generation"""
    latent_dim: int = 4
    """dimensionality of the latent space"""
    sample_interval: int = 400
    """interval between image samples"""
    noise_dim: int = 10
    """latent code dimension for the Bezier GAN generator"""
    bezier_control_pts: int = 32
    """number of control points for the Bezier curve"""


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        layer_width: tuple[int, ...],
        activation_block: Callable = nn.LeakyReLU,
        alpha: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.model = self._build_model(layer_width, activation_block, alpha)

    def _build_model(
        self,
        layer_width: tuple[int, ...],
        activation_block: Callable,
        alpha: float,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_sizes = (self.in_features, *layer_width)
        out_sizes = (*layer_width, self.out_features)
        for idx, (in_f, out_f) in enumerate(zip(in_sizes, out_sizes)):
            layers.append(nn.Linear(in_f, out_f))
            if idx < len(layer_width):  # Hidden layers, not the final one
                layers.append(nn.BatchNorm1d(out_f))
                layers.append(activation_block(alpha))
        return nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward pass for the MLP."""
        return self.model(x)


class Deconv1DCombo(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        alpha: float = 0.2,
    ):
        super().__init__()
        self.seq = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(alpha),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward pass for Deconv1DCombo."""
        return self.seq(x)


class CPWGenerator(nn.Module):
    def __init__(
        self,
        in_features: int,
        n_control_points: int,
        dense_layers: tuple[int, ...] = (1024,),
        deconv_channels: tuple[int, ...] = (96 * 8, 96 * 4, 96 * 2, 96),
    ):
        super().__init__()
        self.in_features = in_features
        self.n_control_points = n_control_points

        self.in_chnl, self.in_width = self._calculate_parameters(n_control_points, deconv_channels)
        self.dense = MLP(in_features, self.in_chnl * self.in_width, dense_layers)
        self.deconv = self._build_deconv(deconv_channels)
        self.cp_gen = nn.Sequential(nn.Conv1d(deconv_channels[-1], 2, 1), nn.Tanh())
        self.w_gen = nn.Sequential(nn.Conv1d(deconv_channels[-1], 1, 1), nn.Sigmoid())

    def _calculate_parameters(self, n_control_points: int, channels: tuple[int, ...]) -> tuple[int, int]:
        n_l = len(channels) - 1
        in_chnl = channels[0]
        in_width = n_control_points // (2**n_l)
        assert in_width >= 4, (  # noqa: PLR2004
            f"Too many deconvolutional layers ({n_l}) for the {self.n_control_points} control points."
        )
        return in_chnl, in_width

    def _build_deconv(self, channels: tuple[int, ...]) -> nn.Sequential:
        deconv_blocks = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            block = Deconv1DCombo(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
            deconv_blocks.append(block)
        return nn.Sequential(*deconv_blocks)

    def forward(self, z: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Forward pass for the CPWGenerator."""
        x = self.dense(z).view(-1, self.in_chnl, self.in_width)  # [N, in_chnl, in_width]
        x = self.deconv(x)  # [N, out_chnl, width_out]
        cp = self.cp_gen(x)  # [N, 2, n_control_points]
        w = self.w_gen(x)  # [N, 1, n_control_points]
        return cp, w


class BezierLayer(nn.Module):
    def __init__(self, in_features: int, n_control_points: int, n_data_points: int, eps: float = 1e-7):
        super().__init__()
        self.in_features = in_features
        self.n_control_points = n_control_points
        self.n_data_points = n_data_points
        self.eps = eps

        self.generate_intervals = nn.Sequential(
            nn.Linear(in_features, n_data_points - 1),
            nn.Softmax(dim=1),
            nn.ConstantPad1d((1, 0), 0),  # leading zero
        )

    def forward(
        self,
        features: th.Tensor,
        control_points: th.Tensor,
        weights: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Forward pass for the Bezier layer."""
        cp, w = self._check_consistency(control_points, weights)
        bs, pv, intvls = self._generate_bernstein_polynomial(features)
        dp = (cp * w) @ bs / (w @ bs)  # [N, 2, n_data_points]
        return dp, pv, intvls

    def _check_consistency(self, control_points: th.Tensor, weights: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        assert control_points.shape[-1] == self.n_control_points
        assert weights.shape[-1] == self.n_control_points
        return control_points, weights

    def _generate_bernstein_polynomial(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        intvls = self.generate_intervals(features)  # [N, n_data_points]
        pv = th.cumsum(intvls, dim=-1).clamp(0, 1).unsqueeze(1)  # [N, 1, n_data_points]

        pw1 = th.arange(0.0, self.n_control_points, device=features.device).view(1, -1, 1)
        pw2 = th.flip(pw1, (1,))
        lbs = (
            pw1 * th.log(pv + self.eps)
            + pw2 * th.log(1 - pv + self.eps)
            + th.lgamma(th.tensor(self.n_control_points, device=features.device) + self.eps).view(1, -1, 1)
            - th.lgamma(pw1 + 1 + self.eps)
            - th.lgamma(pw2 + 1 + self.eps)
        )
        bs = th.exp(lbs)  # [N, n_control_points, n_data_points]
        return bs, pv, intvls


class Generator(nn.Module):
    """Bezier GAN generator.

    this combines c (latent code) + z (noise) and passes them through:
    1) MLP => (features) => intervals
    2) CPW => control points + weights
    3) Bezier => final design
    """

    def __init__(  # noqa: PLR0913
        self,
        latent_dim: int,
        noise_dim: int,
        num_conds: int,
        n_control_points: int,
        n_data_points: int,
        conds_normalizer: Normalizer,
        design_scalars_normalizer: Normalizer,
        eps: float = 1e-7,
        m_features: int = 256,
        feature_gen_layers: tuple[int, ...] = (1024,),
        dense_layers: tuple[int, ...] = (1024,),
        deconv_channels: tuple[int, ...] = (768, 384, 192, 96),
        scalar_features: int = 1,
        scalar_layers: tuple[int, ...] = (128, 128, 128, 128),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        self.conds_normalizer = conds_normalizer
        self.design_scalars_normalizer = design_scalars_normalizer

        # total input dimension for the MLP
        total_in = latent_dim + noise_dim + num_conds
        self.feature_generator = MLP(total_in, m_features, feature_gen_layers)
        self.scalar_generator = MLP(m_features, scalar_features, scalar_layers)
        self.cpw_generator = CPWGenerator(total_in, n_control_points, dense_layers, deconv_channels)
        self.bezier_layer = BezierLayer(m_features, n_control_points, n_data_points, eps)

    def forward(
        self, c: th.Tensor, z: th.Tensor, conds: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Forward pass for the generator.

        :param c: [N, latent_dim]  (sampled from uniform in [bounds[0], bounds[1]])
        :param z: [N, noise_dim]   (sampled from normal(0, 0.5))
        :param conds: [N, num_conds] (design conditions)
        :return: (dp, cp, w, ub, db) => (data points, control points, )
        """
        # Normalize the conditions
        conds = self.conds_normalizer.normalize(conds)
        # 1) Combine c + z
        combined = th.cat([c, z, conds], dim=1)  # shape [N, latent_dim+noise_dim]
        # 2) MLP => features
        features = self.feature_generator(combined)
        # 3) control points + weights
        cp, w = self.cpw_generator(combined)
        # 4) produce final design
        dp, ub, db = self.bezier_layer(features, cp, w)

        # 2.1) features => scalar outputs
        sf = th.sigmoid(self.scalar_generator(features))
        sf = self.design_scalars_normalizer.denormalize(sf)
        return dp, cp, w, ub, db, sf


class Discriminator(nn.Module):
    """Bezier GAN discriminator."""

    def __init__(  # noqa: PLR0913
        self,
        latent_dim: int,
        design_scalars: int,
        num_conds: int,
        design_shape: tuple,
        conds_normalizer: Normalizer,
        design_scalars_normalizer: Normalizer,
        dropout: float = 0.4,
        momentum: float = 0.9,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.conds_normalizer = conds_normalizer
        self.design_scalars_normalizer = design_scalars_normalizer
        # First conv: kernel_size=(2,4), stride=(1,2).
        # This will reduce height from 2 -> 1 (because kernel=2, stride=1)
        # and width from 192 -> ~ (192/2)=96 (depending on padding).
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(2, 4), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(64, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
        )

        # Second conv: kernel_size=(1,4), since now height=1, width~=96
        # stride=(1,2) again, so width will shrink further.
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(128, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
        )

        # Flatten and a small MLP head for D and Q:
        # Use a dummy pass to measure the conv output shape:
        test_in = th.zeros(1, 1, design_shape[0], design_shape[1])
        out = self.conv1(test_in)
        out = self.conv2(out)
        flat_dim = out.numel()

        self.post_conv_fc = nn.Sequential(
            nn.Linear(flat_dim + design_scalars + num_conds, 256),
            nn.BatchNorm1d(256, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # d-head
        self.d_out = nn.Linear(256, 1)
        # q-head
        self.q_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.q_mean = nn.Linear(128, latent_dim)
        self.q_logstd = nn.Linear(128, latent_dim)

    def forward(self, x: th.Tensor, conds: th.Tensor, design_scalars: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Forward pass for the discriminator."""
        # x is [batch_size, 2, 192] => shape => [N, 1, 2, 192]
        x = x.unsqueeze(1)

        out = self.conv1(x)
        out = self.conv2(out)

        out = out.view(out.size(0), -1)
        conds = self.conds_normalizer.normalize(conds)
        design_scalars = self.design_scalars_normalizer.normalize(design_scalars)
        out = th.cat((out, conds, design_scalars), dim=1)
        out = self.post_conv_fc(out)

        d = self.d_out(out)

        q_int = self.q_fc(out)
        q_mean = self.q_mean(q_int)
        q_logstd = self.q_logstd(q_int)
        q_logstd = th.clamp(q_logstd, min=-16)

        # q shape [N, 2, latent_dim]
        q = th.stack([q_mean, q_logstd], dim=1)
        return d, q


def bce_with_logits(pred: th.Tensor, target: th.Tensor) -> th.Tensor:
    """Replicates tensorflow sigmoid_cross_entropy_with_logits."""
    return f.binary_cross_entropy_with_logits(pred, target, reduction="mean")


def compute_r_loss(cp: th.Tensor, w: th.Tensor) -> th.Tensor:
    """Computes regularization loss."""
    r_w_loss = w[:, :, 1:-1].mean()

    cp_diff = cp[:, :, 1:] - cp[:, :, :-1]
    cp_dist = cp_diff.norm(dim=1)
    r_cp_loss = cp_dist.mean()

    ends = cp[:, :, 0] - cp[:, :, -1]
    end_norm = ends.norm(dim=1)
    penal = th.clamp(-10 * ends[:, 1], min=0.0)
    r_ends_loss = end_norm + penal
    r_ends_loss_mean = r_ends_loss.mean()

    return r_w_loss + r_cp_loss + r_ends_loss_mean


def compute_q_loss(q_mean: th.Tensor, q_logstd: th.Tensor, q_target: th.Tensor) -> th.Tensor:
    """Computes latent code inference loss."""
    epsilon = (q_target - q_mean) / (q_logstd.exp() + _EPS)
    q_loss_elem = q_logstd + 0.5 * (epsilon**2)
    return q_loss_elem.mean()


class Normalizer:
    """Normalizes or denormalizes the input tensor."""

    def __init__(self, min_val: th.Tensor, max_val: th.Tensor, eps: float = 1e-7):
        self.eps = eps
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, x: th.Tensor) -> th.Tensor:
        """Normalizes the input tensor."""
        return (x - self.min_val) / (self.max_val - self.min_val + self.eps)

    def denormalize(self, x: th.Tensor) -> th.Tensor:
        """Denormalizes the input tensor."""
        return x * (self.max_val - self.min_val + self.eps) + self.min_val


if __name__ == "__main__":
    args = tyro.cli(Args)
    th.autograd.set_detect_anomaly(True)

    # Setup
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    run_name = f"{args.problem_id}__{args.algo}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            save_code=True,
            name=run_name,
        )

    th.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    th.backends.cudnn.deterministic = True

    os.makedirs("images", exist_ok=True)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    bezier_control_pts = args.bezier_control_pts
    n_data_points = problem.design_space["coords"].shape[1]  # for airfoil, 192

    # The Discriminator uses shape [N, 2, #points].
    problem_dataset = problem.dataset.with_format("torch")["train"]
    design_scalar_keys = list(problem_dataset["optimal_design"][0].keys())
    design_scalar_keys.remove("coords")
    coords_set = [problem_dataset[i]["optimal_design"]["coords"][:] for i in range(len(problem_dataset))]
    design_scalars = [example["optimal_design"][key] for example in problem_dataset for key in design_scalar_keys]
    training_ds = th.utils.data.TensorDataset(
        th.stack(coords_set),
        th.stack(design_scalars).unsqueeze(1),
        *[problem_dataset[key][:] for key, _ in problem.conditions],
    )

    cond_tensors = th.stack(training_ds.tensors[2:])
    conds_min = cond_tensors.amin(dim=tuple(range(1, cond_tensors.ndim))).to(device)
    conds_max = cond_tensors.amax(dim=tuple(range(1, cond_tensors.ndim))).to(device)
    design_scalars_min = training_ds.tensors[1].amin(dim=0).to(device)
    design_scalars_max = training_ds.tensors[1].amax(dim=0).to(device)

    conds_normalizer = Normalizer(conds_min, conds_max)
    design_scalars_normalizer = Normalizer(design_scalars_min, design_scalars_max)

    dataloader = th.utils.data.DataLoader(
        training_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    discriminator = Discriminator(
        latent_dim=args.latent_dim,
        design_scalars=len(design_scalar_keys),
        num_conds=len(problem.conditions),
        design_shape=problem.design_space["coords"].shape,
        conds_normalizer=conds_normalizer,
        design_scalars_normalizer=design_scalars_normalizer,
    ).to(device)

    generator = Generator(
        latent_dim=args.latent_dim,
        noise_dim=args.noise_dim,
        num_conds=len(problem.conditions),
        n_control_points=bezier_control_pts,
        n_data_points=n_data_points,
        conds_normalizer=conds_normalizer,
        design_scalars_normalizer=design_scalars_normalizer,
        eps=_EPS,
        scalar_features=1,
    ).to(device)

    # Two separate Adam optimizers
    d_optimizer = th.optim.Adam(discriminator.parameters(), lr=args.lr_disc, betas=(args.b1, args.b2), eps=_EPS)
    g_optimizer = th.optim.Adam(generator.parameters(), lr=args.lr_gen, betas=(args.b1, args.b2), eps=_EPS)

    # Bounds for c
    bounds = (0.0, 1.0)

    @th.no_grad()
    def sample_designs(n_designs: int) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Samples n_designs from the generator."""
        # Sample noise
        c = (bounds[1] - bounds[0]) * th.rand(n_designs, args.latent_dim, device=device) + bounds[0]
        z = 0.5 * th.randn(n_designs, args.noise_dim, device=device)
        steps = th.linspace(0, 1, n_designs, device=device).view(n_designs, 1, 1)
        conds = conds_min + steps * (conds_max - conds_min)
        dp, _, _, _, _, sf = generator(c, z, conds.squeeze(1))
        return conds, dp, sf

    for epoch in range(args.n_epochs):
        for i, real_designs_cpu in enumerate(dataloader):
            real_designs = real_designs_cpu[0].to(device)
            real_alpha = real_designs_cpu[1].to(device)
            real_conds = th.stack(real_designs_cpu[2:]).T.to(device)

            # ===== Step 1: D train real =====
            d_optimizer.zero_grad()
            logits_real, _ = discriminator(real_designs, real_conds, real_alpha)
            d_loss_real = bce_with_logits(logits_real, th.ones_like(logits_real, device=device))
            d_loss_real.backward()

            d_optimizer.step()

            # ===== Step 2: D train fake + Q =====
            c = (bounds[1] - bounds[0]) * th.rand(real_conds.shape[0], args.latent_dim, device=device) + bounds[0]
            z = 0.5 * th.randn(real_conds.shape[0], args.noise_dim, device=device)
            d_optimizer.zero_grad()

            x_fake, cp, w, ub, db, sf = generator(c, z, real_conds)
            logits_fake, q_out = discriminator(x_fake, real_conds, sf)

            d_loss_fake = bce_with_logits(logits_fake, th.zeros_like(logits_fake, device=device))

            q_mean = q_out[:, 0, :]
            q_logstd = q_out[:, 1, :]
            q_loss = compute_q_loss(q_mean, q_logstd, q_target=c)
            d_loss = d_loss_fake + q_loss
            d_loss.backward()
            d_optimizer.step()

            # ===== Step 3: G train (g_loss + 10*r_loss + q_loss) =====
            c2 = (bounds[1] - bounds[0]) * th.rand(real_conds.shape[0], args.latent_dim, device=device) + bounds[0]
            z2 = 0.5 * th.randn(real_conds.shape[0], args.noise_dim, device=device)
            x_fake2, cp2, w2, ub2, _, sf2 = generator(c2, z2, real_conds)

            logits_fake2, q_out2 = discriminator(x_fake2, real_conds, sf2)
            g_loss_base = bce_with_logits(logits_fake2, th.ones_like(logits_fake2, device=device))
            r_loss = compute_r_loss(cp2, w2)

            q_mean2 = q_out2[:, 0, :]
            q_logstd2 = q_out2[:, 1, :]
            q_loss2 = compute_q_loss(q_mean2, q_logstd2, q_target=c2)
            total_g_loss = g_loss_base + 10 * r_loss + q_loss2
            g_optimizer.zero_grad()
            total_g_loss.backward()
            g_optimizer.step()

            # ----- Logging / printing -----
            batches_done = epoch * len(dataloader) + i
            if args.track:
                wandb.log(
                    {
                        "d_loss": d_loss.item(),
                        "g_loss": g_loss_base.item(),
                        "epoch": epoch,
                        "batch": batches_done,
                    }
                )

            # Save images periodically
            if batches_done % args.sample_interval == 0:
                print(
                    f"[Epoch {epoch}/{args.n_epochs}] "
                    f"[Batch {i}/{len(dataloader)}] "
                    f"[d_loss_fake: {d_loss_fake.item():.6f}] "
                    f"[d_loss_real: {d_loss_real.item():.6f}] "
                    f"[q_loss: {q_loss.item():.6f}] "
                    f"[r_loss: {r_loss.item():.6f}] "
                    f"[q_loss2: {q_loss2.item():.6f}] "
                    f"[g_loss_base: {g_loss_base.item():.6f}]"
                )

                conds, dp, sf = sample_designs(25)
                do = th.cat((sf, conds.squeeze(1)), dim=1).cpu().numpy()
                fig, axes = plt.subplots(5, 5, figsize=(20, 20), dpi=300)
                axes = axes.flatten()
                for j in range(25):
                    do1 = do[j]
                    coords = dp[j].cpu().numpy()  # [2, #points]
                    design = {"coords": coords, "angle_of_attack": do1[0]}
                    fig, ax = problem.render(design)
                    ax.figure.canvas.draw()
                    img = np.array(fig.canvas.renderer.buffer_rgba())
                    axes[j].imshow(img)
                    title = [(problem.conditions[i - 1][0], f"{do1[i]:.2f}") for i in range(1, len(do1))]
                    title_string = "\n ".join(f"{condition}: {value}" for condition, value in title)
                    axes[j].title.set_text(title_string)  # Set title
                    axes[j].set_xticks([])
                    axes[j].set_yticks([])
                plt.tight_layout()
                img_fname = f"images/{batches_done}.png"
                plt.savefig(img_fname)
                plt.close()

                if args.track:
                    wandb.log({"designs": wandb.Image(img_fname)})

            # --------------
            #  Save models
            # --------------
            if args.save_model and epoch == args.n_epochs - 1 and i == len(dataloader) - 1:
                ckpt_gen = {
                    "epoch": epoch,
                    "batches_done": batches_done,
                    "generator": generator.state_dict(),
                    "optimizer_generator": g_optimizer.state_dict(),
                    "loss": g_loss_base.item(),
                }
                ckpt_disc = {
                    "epoch": epoch,
                    "batches_done": batches_done,
                    "discriminator": discriminator.state_dict(),
                    "optimizer_discriminator": d_optimizer.state_dict(),
                    "loss": d_loss.item(),
                }

                th.save(ckpt_gen, "bezier_generator.pth")
                th.save(ckpt_disc, "bezier_discriminator.pth")
                if args.track:
                    artifact_gen = wandb.Artifact(f"{args.problem_id}_{args.algo}_generator", type="model")
                    artifact_gen.add_file("bezier_generator.pth")
                    artifact_disc = wandb.Artifact(f"{args.problem_id}_{args.algo}_discriminator", type="model")
                    artifact_disc.add_file("bezier_discriminator.pth")

                    wandb.log_artifact(artifact_gen, aliases=[f"seed_{args.seed}"])
                    wandb.log_artifact(artifact_disc, aliases=[f"seed_{args.seed}"])

    if args.track:
        wandb.finish()
