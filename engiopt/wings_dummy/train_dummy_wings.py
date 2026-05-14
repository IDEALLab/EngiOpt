from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
import torch as th
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import tyro
import wandb


@dataclass
class Args:
    # logging
    track: bool = True
    wandb_project: str = "engiopt"
    wandb_entity: str | None = None
    seed: int = 1

    # data
    n_train: int = 2048
    n_val: int = 512
    batch_size: int = 64

    # training
    n_epochs: int = 10
    lr: float = 1e-3

    # shape: [B, 9, 2, 192]
    n_slices: int = 9
    n_coords: int = 2
    n_points: int = 192

    # output dim (keep 1 for first pass)
    y_dim: int = 1


def make_dummy_dataset(n: int, n_slices: int, n_coords: int, n_points: int, y_dim: int, seed: int) -> TensorDataset:
    """Dummy X: random wing tensors. Dummy y: mean(x) + tiny noise (learnable signal)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, n_slices, n_coords, n_points), dtype=np.float32)

    y = x.mean(axis=(1, 2, 3)).astype(np.float32)  # [n]
    y += 0.01 * rng.standard_normal((n,), dtype=np.float32)  # small noise
    y = y.reshape(n, 1)
    if y_dim != 1:
        y = np.repeat(y, y_dim, axis=1)

    return TensorDataset(th.from_numpy(x), th.from_numpy(y))


class SliceEncoder(nn.Module):
    """Encodes one slice shaped [B, 2, 192] -> [B, d]."""
    def __init__(self, d: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, d, kernel_size=5, padding=2),
            nn.ReLU(),
        )

    def forward(self, x_slice: th.Tensor) -> th.Tensor:
        # x_slice: [B, 2, 192]
        h = self.net(x_slice)              # [B, d, 192]
        h = h.mean(dim=-1)                 # global avg pool over points -> [B, d]
        return h


class DummyWingModel(nn.Module):
    """Input [B, 9, 2, 192] -> output [B, y_dim]."""
    def __init__(self, y_dim: int = 1, d: int = 64):
        super().__init__()
        self.slice_encoder = SliceEncoder(d=d)
        self.head = nn.Sequential(
            nn.Linear(d, 64),
            nn.ReLU(),
            nn.Linear(64, y_dim),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x: [B, 9, 2, 192]
        b, s, c, p = x.shape
        x = x.view(b * s, c, p)            # [B*9, 2, 192]
        z = self.slice_encoder(x)          # [B*9, d]
        z = z.view(b, s, -1)               # [B, 9, d]
        wing_feat = z.mean(dim=1)          # mean over slices -> [B, d]
        return self.head(wing_feat)        # [B, y_dim]


def main():
    args = tyro.cli(Args)

    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = th.device("mps" if th.backends.mps.is_available() else ("cuda" if th.cuda.is_available() else "cpu"))
    print("device:", device)

    run_name = f"dummy_wings__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), name=run_name)

    train_ds = make_dummy_dataset(args.n_train, args.n_slices, args.n_coords, args.n_points, args.y_dim, seed=args.seed)
    val_ds = make_dummy_dataset(args.n_val, args.n_slices, args.n_coords, args.n_points, args.y_dim, seed=args.seed + 123)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = DummyWingModel(y_dim=args.y_dim, d=64).to(device)
    opt = th.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    global_step = 0
    for epoch in range(args.n_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if args.track:
                wandb.log({"train/loss": float(loss.item()), "epoch": epoch}, step=global_step)
            global_step += 1

        # validation
        model.eval()
        vals = []
        with th.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                vals.append(loss_fn(pred, yb).item())
        val_loss = float(np.mean(vals))

        print(f"epoch {epoch}: val_loss={val_loss:.6f}")
        if args.track:
            wandb.log({"val/loss": val_loss, "epoch": epoch}, step=global_step)

    if args.track:
        wandb.finish()


if __name__ == "__main__":
    main()