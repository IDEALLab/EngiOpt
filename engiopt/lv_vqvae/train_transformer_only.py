"""Train Stage 2 (Transformer) from a pre-trained Stage 1 (LV-VQVAE) checkpoint."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import os
import random
import time
from typing import Any

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as f
import tqdm
import tyro
import wandb

from engiopt.lv_vqvae.lv_vqvae import VQVAE
from engiopt.lv_vqvae.lv_vqvae import VQVAETransformer
from engiopt.transforms import drop_constant
from engiopt.transforms import normalize
from engiopt.transforms import resize_to


@dataclass
class TransformerOnlyArgs:
    """Command-line arguments for Transformer-Only training."""

    # Checkpoint to load
    stage1_wandb_run_path: str
    """Path to the wandb run containing the Stage 1 artifacts (e.g., 'adrake17/engiopt/v1x2y3z')"""

    problem_id: str = "beams2d"
    algo: str = "transformer_only"

    track: bool = True
    wandb_project: str = "engiopt"
    wandb_entity: str | None = None
    seed: int = 1
    save_model: bool = True

    conditional: bool = True
    normalize_conditions: bool = True
    drop_constant_conditions: bool = True
    image_size: int = 128

    # Stage 2 (Transformer) Hyperparams
    n_epochs_transformer: int = 100
    early_stopping: bool = True
    early_stopping_patience: int = 3
    early_stopping_delta: float = 1e-3

    batch_size_transformer: int = 32
    lr_transformer: float = 6e-4
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.3
    sample_interval_transformer: int = 100


if __name__ == "__main__":
    args = tyro.cli(TransformerOnlyArgs)

    # Seeding
    th.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    th.backends.cudnn.deterministic = True
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)
    design_shape = problem.design_space.shape

    # Configure data loader
    training_ds = problem.dataset.with_format("torch")["train"]
    len_dataset = len(training_ds)

    training_ds = training_ds.map(
        lambda batch: {
            "optimal_upsampled": resize_to(data=batch["optimal_design"][:], h=args.image_size, w=args.image_size)
            .cpu()
            .numpy()
        },
        batched=True,
    ).remove_columns("optimal_design")

    conditions = problem.conditions_keys
    if args.drop_constant_conditions:
        training_ds, conditions = drop_constant(training_ds, conditions)

    mean: Any = None
    std: Any = None
    if args.normalize_conditions:
        training_ds, mean, std = normalize(training_ds, conditions)

    n_conds = len(conditions)
    condition_tensors = [training_ds[key][:] for key in conditions]

    th_training_ds = th.utils.data.TensorDataset(
        th.as_tensor(training_ds["optimal_upsampled"][:]).to(device),
        *[th.as_tensor(training_ds[key][:]).to(device) for key in conditions],
    )
    dataloader_transformer = th.utils.data.DataLoader(th_training_ds, batch_size=args.batch_size_transformer, shuffle=True)

    # Validation Dataset
    val_ds = problem.dataset.with_format("torch")["val"]
    val_ds = val_ds.map(
        lambda batch: {
            "optimal_upsampled": resize_to(data=batch["optimal_design"][:], h=args.image_size, w=args.image_size)
            .cpu()
            .numpy()
        },
        batched=True,
    ).remove_columns("optimal_design")

    if args.drop_constant_conditions:
        to_drop = [c for c in problem.conditions_keys if c not in conditions]
        if to_drop:
            val_ds = val_ds.remove_columns(to_drop)

    if args.normalize_conditions:
        val_ds = val_ds.map(
            lambda batch: {
                c: ((th.as_tensor(batch[c][:]).float() - mean[i]) / std[i]).numpy() for i, c in enumerate(conditions)
            },
            batched=True,
        )

    th_val_ds = th.utils.data.TensorDataset(
        th.as_tensor(val_ds["optimal_upsampled"][:]).to(device),
        *[th.as_tensor(val_ds[key][:]).to(device) for key in conditions],
    )
    dataloader_val = th.utils.data.DataLoader(th_val_ds, batch_size=args.batch_size_transformer, shuffle=False)

    n_logged_designs = 16
    fixed_indices = random.sample(range(len(th_val_ds)), n_logged_designs)
    log_subset = th.utils.data.Subset(th_val_ds, fixed_indices)
    log_dataloader = th.utils.data.DataLoader(log_subset, batch_size=n_logged_designs, shuffle=False)

    # Logging
    run_name = f"{args.problem_id}__{args.algo}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
        )
        wandb.define_metric("transformer_step", summary="max")
        wandb.define_metric("transformer_loss", step_metric="transformer_step")
        wandb.define_metric("epoch_transformer", step_metric="transformer_step")
        # Ensure this is defined regardless of early stopping status since we always log it now
        wandb.define_metric("transformer_val_loss", step_metric="transformer_step")

    # =========================================================================
    # LOAD STAGE 1 MODELS FROM WANDB
    # =========================================================================
    print(f"Loading Stage 1 models from: {args.stage1_wandb_run_path}")
    api = wandb.Api()
    run = api.run(args.stage1_wandb_run_path)

    image_channels = run.config["image_channels"]
    latent_size = run.config["latent_size"]

    artifact_cvqvae = api.artifact(
        f"{run.entity}/{run.project}/{args.problem_id}_lv_vqvae_lv_cvqvae:latest",
        type="model",
    )
    artifact_vqvae = api.artifact(
        f"{run.entity}/{run.project}/{args.problem_id}_lv_vqvae_lv_vqvae:latest",
        type="model",
    )

    ckpt_path_cvqvae = os.path.join(artifact_cvqvae.download(), "lv_cvqvae.pth")
    ckpt_path_vqvae = os.path.join(artifact_vqvae.download(), "lv_vqvae.pth")

    ckpt_cvqvae = th.load(ckpt_path_cvqvae, map_location=device, weights_only=False)
    ckpt_vqvae = th.load(ckpt_path_vqvae, map_location=device, weights_only=False)

    vqvae = VQVAE(
        device=device,
        is_c=False,
        use_vq=True,
        encoder_channels=run.config["encoder_channels"],
        encoder_start_resolution=run.config["image_size"],
        encoder_attn_resolutions=run.config["encoder_attn_resolutions"],
        encoder_num_res_blocks=run.config["encoder_num_res_blocks"],
        decoder_channels=run.config["decoder_channels"],
        decoder_start_resolution=run.config["latent_size"],
        decoder_num_res_blocks=run.config["decoder_num_res_blocks"],
        image_channels=image_channels,
        latent_dim=run.config["latent_dim"],
        num_codebook_vectors=run.config["num_codebook_vectors"],
    ).to(device)
    vqvae.load_state_dict(ckpt_vqvae["vqvae"])
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)  # noqa: FBT003

    cvqvae = VQVAE(
        device=device,
        is_c=True,
        use_vq=True,
        cond_feature_map_dim=run.config["cond_feature_map_dim"],
        cond_dim=run.config["cond_dim"],
        cond_hidden_dim=run.config["cond_hidden_dim"],
        cond_latent_dim=run.config["cond_latent_dim"],
        cond_codebook_vectors=run.config["cond_codebook_vectors"],
    ).to(device)
    cvqvae.load_state_dict(ckpt_cvqvae["cvqvae"])
    cvqvae.eval()
    for p in cvqvae.parameters():
        p.requires_grad_(False)  # noqa: FBT003

    lv_state = ckpt_vqvae.get("lv_state", {})
    active_mask = lv_state.get("active_mask", None)
    frozen_mean = lv_state.get("frozen_mean", None)
    if active_mask is not None:
        active_mask = active_mask.to(device, dtype=th.bool)
    if frozen_mean is not None:
        frozen_mean = frozen_mean.to(device, dtype=th.float32)

    # =========================================================================
    # INITIALIZE STAGE 2 TRANSFORMER
    # =========================================================================
    transformer = VQVAETransformer(
        conditional=args.conditional,
        vqvae=vqvae,
        cvqvae=cvqvae,
        image_size=args.image_size,
        decoder_channels=run.config["decoder_channels"],
        cond_feature_map_dim=run.config["cond_feature_map_dim"],
        num_codebook_vectors=run.config["num_codebook_vectors"],
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    ).to(device)

    transformer.active_mask = active_mask
    transformer.frozen_mean = frozen_mean
    transformer.train()

    decay: set[str] = set()
    no_decay: set[str] = set()
    for mn, m in transformer.transformer.named_modules():
        for pn, _ in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn
            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, nn.Linear):
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, (nn.LayerNorm, nn.Embedding)):
                no_decay.add(fpn)

    no_decay.add("pos_emb")
    param_dict = dict(transformer.transformer.named_parameters())
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(decay & set(param_dict.keys()))],
            "weight_decay": 0.01,
        },
        {
            "params": [param_dict[pn] for pn in sorted(no_decay & set(param_dict.keys()))],
            "weight_decay": 0.0,
        },
    ]
    opt_transformer = th.optim.AdamW(optim_groups, lr=args.lr_transformer, betas=(0.9, 0.95))

    @th.no_grad()
    def sample_designs_transformer(n_designs: int) -> tuple[th.Tensor, th.Tensor]:  # noqa: D103
        transformer.eval()
        all_conditions = th.stack(condition_tensors, dim=1)
        linspaces = [
            th.linspace(
                all_conditions[:, i].min(),
                all_conditions[:, i].max(),
                n_designs,
                device=device,
            )
            for i in range(all_conditions.shape[1])
        ]
        desired_conds = th.stack(linspaces, dim=1)

        if args.conditional:
            c = transformer.encode_to_z(x=desired_conds, is_c=True)[1]
        else:
            c = th.ones(n_designs, 1, dtype=th.int64, device=device) * transformer.sos_token

        # PURE MULTINOMIAL SAMPLING (Original behavior)
        latent_imgs = transformer.sample(
            x=th.empty(n_designs, 0, dtype=th.int64, device=device),
            c=c,
            steps=(latent_size**2),
        )
        gen_imgs = transformer.z_to_image(latent_imgs)
        transformer.train()
        return desired_conds, gen_imgs

    # =========================================================================
    # TRAIN LOOP
    # =========================================================================
    print("Stage 2: Training Transformer")
    val_loss: float = float("nan")

    if args.early_stopping:
        best_val: float = float("inf")
        best_ckpt_tr: dict[str, Any] | None = None
        patience_counter: int = 0

    for epoch in tqdm.trange(args.n_epochs_transformer):
        for i, data in enumerate(dataloader_transformer):
            designs = data[0].to(dtype=th.float32, device=device)
            conds = th.stack((data[1:]), dim=1).to(dtype=th.float32, device=device)

            opt_transformer.zero_grad()
            logits, targets = transformer(designs, conds)
            loss = f.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            with th.no_grad():
                probs = f.softmax(logits, dim=-1)
                clamped_probs = probs.clamp_min(1e-12)
                entropy = -(clamped_probs * clamped_probs.log()).sum(dim=-1)
                transformer_logits_entropy = float(entropy.mean().item())

            loss.backward()
            opt_transformer.step()

            if args.track:
                batches_done = epoch * len(dataloader_transformer) + i

                # 1. Start a single dictionary for this step
                step_logs = {
                    "transformer_loss": loss.item(),
                    "transformer_logits_entropy": transformer_logits_entropy,
                    "transformer_step": batches_done,
                    "epoch_transformer": epoch,
                }

                print(
                    f"[Epoch {epoch}/{args.n_epochs_transformer}] "
                    f"[Batch {i}/{len(dataloader_transformer)}] "
                    f"[Loss: {loss.item()}] "
                    f"[Entropy: {transformer_logits_entropy:.2f}]"
                )

                # 2. Only run the expensive sampling if it's time
                if batches_done % args.sample_interval_transformer == 0:
                    desired_conds, gen_designs = sample_designs_transformer(n_designs=n_logged_designs)

                    if args.normalize_conditions and std is not None and mean is not None:
                        desired_conds = (desired_conds.cpu() * std) + mean

                    gen_designs = resize_to(data=gen_designs, h=design_shape[0], w=design_shape[1])

                    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
                    axes = axes.flatten()
                    for j, tensor in enumerate(gen_designs):
                        img = tensor.cpu().numpy().reshape(design_shape[0], design_shape[1])
                        dc = desired_conds[j].cpu()
                        axes[j].imshow(img)
                        title = [(conditions[idx], f"{dc[idx]:.2f}") for idx in range(n_conds)]
                        title_string = "\n ".join(f"{condition}: {value}" for condition, value in title)
                        axes[j].set_title(title_string, fontsize=8)
                        axes[j].axis("off")

                    plt.tight_layout()

                    # Add the image to the same dictionary
                    step_logs["designs_transformer"] = wandb.Image(fig)
                    plt.close(fig)

                # 3. Log step dict
                wandb.log(step_logs)

        # =====================================================================
        # VALIDATION & EARLY STOPPING (Uncoupled)
        # =====================================================================
        if args.track:
            transformer.eval()
            val_losses = []
            with th.no_grad():
                for val_data in dataloader_val:
                    val_designs = val_data[0].to(dtype=th.float32, device=device)
                    val_conds = th.stack((val_data[1:]), dim=1).to(dtype=th.float32, device=device)
                    val_logits, val_targets = transformer(val_designs, val_conds)
                    v_loss = f.cross_entropy(
                        val_logits.reshape(-1, val_logits.size(-1)),
                        val_targets.reshape(-1),
                    )
                    val_losses.append(v_loss.item())

            val_loss = sum(val_losses) / len(val_losses)

            # ALWAYS log validation loss
            wandb.log({"transformer_val_loss": val_loss, "transformer_step": batches_done})

            # ONLY execute early stopping behavior if enabled
            if args.early_stopping:
                if val_loss < best_val - args.early_stopping_delta:
                    best_val = val_loss
                    patience_counter = 0
                    if args.save_model:
                        best_ckpt_tr = {
                            "epoch": epoch,
                            "batches_done": batches_done,
                            "transformer": copy.deepcopy(transformer.state_dict()),
                            "optimizer_transformer": copy.deepcopy(opt_transformer.state_dict()),
                            "loss": loss.item(),
                            "val_loss": val_loss,
                        }
                else:
                    patience_counter += 1
                    if patience_counter >= args.early_stopping_patience:
                        print(f"Early stopping at epoch {epoch} | best val loss: {best_val:.6f}")
                        break
            transformer.train()

    # Save Final
    if args.track and args.save_model:
        ckpt_tr = (
            best_ckpt_tr
            if (args.early_stopping and best_ckpt_tr is not None)
            else {
                "epoch": epoch,
                "batches_done": batches_done,
                "transformer": transformer.state_dict(),
                "optimizer_transformer": opt_transformer.state_dict(),
                "loss": loss.item(),
                "val_loss": val_loss,  # Uses the final computed val_loss reliably
            }
        )
        th.save(ckpt_tr, "transformer.pth")
        artifact_tr = wandb.Artifact(f"{args.problem_id}_transformer_only", type="model")
        artifact_tr.add_file("transformer.pth")
        wandb.log_artifact(artifact_tr, aliases=[f"seed_{args.seed}"])

    wandb.finish()
